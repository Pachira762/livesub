use std::sync::Arc;

use candle::Result;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};

pub enum MelFilterType {
    Librosa,
    Kaldi,
}

pub struct MelProcessor {
    preemph: f32,
    window: Vec<f32>,
    hop_len: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    spectrum: Vec<f32>,
    n_mels: usize,
    mel_filter_bank: Vec<(usize, Vec<f32>)>,
    samples: Vec<f32>,
    mels: Vec<f32>,
}

impl MelProcessor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        preemph: f32,
        window: Vec<f32>,
        hop_len: usize,
        n_fft: usize,
        mel_filter_type: MelFilterType,
        n_mels: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
    ) -> Result<Self> {
        let mut fft_planner = RealFftPlanner::<f32>::new();
        let fft = fft_planner.plan_fft_forward(n_fft);
        let fft_in = fft.make_input_vec();
        let fft_out = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();

        let mel_filter_bank =
            generate_mel_filter_bank(mel_filter_type, n_fft, n_mels, sample_rate as _, fmin, fmax);

        Ok(Self {
            preemph,
            window,
            hop_len,
            fft,
            fft_in,
            fft_out,
            fft_scratch,
            spectrum: vec![0.0; n_fft / 2 + 1],
            mel_filter_bank,
            n_mels,
            samples: vec![],
            mels: vec![],
        })
    }

    pub fn push(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);

        if self.samples.len() < self.window.len() {
            return;
        }

        let n_frames = (self.samples.len() - self.window.len()) / self.hop_len + 1;
        for i in 0..n_frames {
            self.process_frame(i);
        }

        let n_hop = n_frames * self.hop_len;
        self.samples.copy_within(n_hop.., 0);
        self.samples.resize(self.samples.len() - n_hop, 0.0);
    }

    fn process_frame(&mut self, i_frame: usize) {
        let beg = i_frame * self.hop_len;
        let end = beg + self.window.len();
        let samples = &self.samples[beg..end];

        let mut prev = samples[0];
        for (fft_in, (&window, &x)) in self.fft_in.iter_mut().zip(self.window.iter().zip(samples)) {
            *fft_in = window * (x - self.preemph * prev);
            prev = x;
        }

        self.fft
            .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch)
            .unwrap();

        for (spec, c) in self.spectrum.iter_mut().zip(&self.fft_out) {
            *spec = c.norm_sqr();
        }

        self.mels
            .extend(self.mel_filter_bank.iter().map(|(offset, coeffs)| {
                coeffs
                    .iter()
                    .zip(&self.spectrum[*offset..])
                    .map(|(&coeff, &spec)| coeff * spec)
                    .sum::<f32>()
                    .max(f32::EPSILON)
                    .ln()
            }));
    }

    pub fn mels(&self) -> &[f32] {
        &self.mels
    }

    pub fn num_mel_bins(&self) -> usize {
        self.n_mels
    }

    pub fn num_frames(&self) -> usize {
        self.mels.len() / self.n_mels
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.mels.clear();
    }
}

fn generate_mel_filter_bank(
    mel_filter_type: MelFilterType,
    n_fft: usize,
    n_mels: usize,
    sr: f32,
    fmin: f32,
    fmax: f32,
) -> Vec<(usize, Vec<f32>)> {
    match mel_filter_type {
        MelFilterType::Librosa => generate_mel_filter_bank_librosa(n_fft, n_mels, sr, fmin, fmax),
        MelFilterType::Kaldi => generate_mel_filter_bank_kaldi(n_fft, n_mels, sr, fmin, fmax),
    }
}

fn generate_mel_filter_bank_librosa(
    n_fft: usize,
    n_mels: usize,
    sr: f32,
    fmin: f32,
    fmax: f32,
) -> Vec<(usize, Vec<f32>)> {
    let n_fft_bins = 1 + n_fft / 2;

    let mel_min = fmin.to_mel(false);
    let mel_max = fmax.to_mel(false);

    let points: Vec<_> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel.to_hertz(false)
        })
        .collect();

    let mut melbank = vec![(0, vec![]); n_mels];
    for i in 0..n_mels {
        let hz_left = points[i];
        let hz_center = points[i + 1];
        let hz_right = points[i + 2];
        let enorm = 2.0 / (hz_right - hz_left);

        for j in 0..n_fft_bins {
            let hz = j as f32 * sr / n_fft as f32;
            if hz > hz_left && hz < hz_right {
                if melbank[i].0 == 0 {
                    melbank[i].0 = j;
                }

                let coeff = if hz < hz_center {
                    (hz - hz_left) / (hz_center - hz_left)
                } else {
                    (hz_right - hz) / (hz_right - hz_center)
                };

                melbank[i].1.push(enorm * coeff);
            }
        }
    }

    melbank
}

fn generate_mel_filter_bank_kaldi(
    n_fft: usize,
    n_mels: usize,
    sr: f32,
    fmin: f32,
    fmax: f32,
) -> Vec<(usize, Vec<f32>)> {
    let n_fft_bins = 1 + n_fft / 2;

    let mel_min = fmin.to_mel(true);
    let mel_max = if fmax > 0.0 { fmax } else { sr / 2.0 + fmax }.to_mel(true);

    let points: Vec<_> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let mut melbank = vec![(0, vec![]); n_mels];

    for i in 0..n_mels {
        let mel_left = points[i];
        let mel_center = points[i + 1];
        let mel_right = points[i + 2];

        for j in 0..n_fft_bins {
            let mel = (j as f32 * sr / n_fft as f32).to_mel(true);
            if mel > mel_left && mel < mel_right {
                if melbank[i].0 == 0 {
                    melbank[i].0 = j;
                }

                let coeff = if mel < mel_center {
                    (mel - mel_left) / (mel_center - mel_left)
                } else {
                    (mel_right - mel) / (mel_right - mel_center)
                };

                melbank[i].1.push(coeff);
            }
        }
    }

    melbank
}

trait Mel {
    fn to_mel(self, htk: bool) -> Self;
    fn to_hertz(self, htk: bool) -> Self;
}

impl Mel for f32 {
    fn to_mel(self, htk: bool) -> Self {
        if htk {
            1127.0 * (1.0 + self / 700.0).ln()
        } else {
            let f_min = 0.0;
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = (min_log_hz - f_min) / f_sp;
            let logstep = (6.4f32).ln() / 27.0;

            if self >= min_log_hz {
                min_log_mel + (self / min_log_hz).ln() / logstep
            } else {
                (self - f_min) / f_sp
            }
        }
    }

    fn to_hertz(self, htk: bool) -> Self {
        if htk {
            700.0 * ((self / 1127.0).exp() - 1.0)
        } else {
            let f_min = 0.0;
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = (min_log_hz - f_min) / f_sp;
            let logstep = (6.4f32).ln() / 27.0;

            if self >= min_log_mel {
                min_log_hz * (logstep * (self - min_log_mel)).exp()
            } else {
                f_min + f_sp * self
            }
        }
    }
}
