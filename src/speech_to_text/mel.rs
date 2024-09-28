use std::{f32::consts::PI, sync::Arc};

use anyhow::Result;
use candle_transformers::models::whisper::{HOP_LENGTH as N_HOP, N_FFT, N_FRAMES};
use rustfft::{num_complex::Complex32 as Complex, Fft, FftPlanner};

const N_FILTER: usize = (N_FFT / 2) + 1;
const MEL_ZERO: f32 = (-10.0 + 4.0) / 4.0;

pub struct MelSpectrogram {
    samples: Vec<f32>,
    mel: Vec<f32>,
    i_frame: usize,
    n_bins: usize,

    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_io: Vec<Complex>,
    fft_scratch: Vec<Complex>,
    magnitude: Vec<f32>,
    filter: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new(num_mel_bins: usize) -> Result<Self> {
        let window = (0..N_FFT)
            .map(|i| 0.5 * (1. - ((2.0 * PI * i as f32) / (N_FFT - 1) as f32).cos()))
            .collect();

        let fft = FftPlanner::new().plan_fft_forward(N_FFT);
        let n_scratch = fft.get_inplace_scratch_len();

        let filter = {
            let mel_bytes = match num_mel_bins {
                80 => include_bytes!("melfilters.bytes").as_slice(),
                128 => include_bytes!("melfilters128.bytes").as_slice(),
                n => anyhow::bail!("unexpected num_mel_bins {n}"),
            };

            let mut filter = vec![0f32; mel_bytes.len() / 4];
            <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
                mel_bytes,
                &mut filter,
            );

            filter
        };

        Ok(Self {
            samples: vec![],
            mel: vec![0.0; num_mel_bins * N_FRAMES],
            i_frame: 0,
            n_bins: num_mel_bins,
            window,
            fft,
            fft_io: vec![Complex::default(); N_FFT],
            fft_scratch: vec![Complex::default(); n_scratch],
            magnitude: vec![0f32; N_FILTER],
            filter,
        })
    }

    pub fn decode(&mut self, samples: &[f32]) -> Option<(&[f32], bool)> {
        self.samples.extend_from_slice(samples);

        let is_new_segment = self.i_frame == 0;
        if is_new_segment {
            self.mel.fill(MEL_ZERO);
        }

        let n_frames = {
            let n_samples = self.samples.len();
            let n_frames = n_samples.saturating_sub(N_FFT - N_HOP) / N_HOP;
            n_frames.min(N_FRAMES) - self.i_frame
        };

        if n_frames == 0 {
            return None;
        }

        for _ in 0..n_frames {
            self.pcm_to_mel();
            self.i_frame += 1;
        }

        if self.i_frame >= N_FRAMES {
            _ = self.samples.drain(..self.i_frame * N_HOP);
            self.i_frame = 0;
        }

        Some((&self.mel, is_new_segment))
    }

    fn pcm_to_mel(&mut self) {
        let i_frame = self.i_frame;
        let mel = &mut self.mel;
        let samples = &self.samples[i_frame * N_HOP..i_frame * N_HOP + N_FFT];

        for (i, io) in self.fft_io.iter_mut().enumerate() {
            io.re = self.window[i] * samples[i];
            io.im = 0.0;
        }

        self.fft
            .process_with_scratch(&mut self.fft_io, &mut self.fft_scratch);

        for (io, magnitude) in self.fft_io.iter().zip(&mut self.magnitude) {
            *magnitude = io.norm_sqr();
        }

        let mut m_max = 0.0;

        for i in 0..self.n_bins {
            let mut m = 0.0;

            for j in 0..N_FILTER {
                m += self.filter[i * N_FILTER + j] * self.magnitude[j];
            }

            m = m.max(1e-10).log10();
            mel[i * N_FRAMES + i_frame] = m;

            m_max = m.max(m_max);
        }

        m_max -= 8.0;

        for i in 0..self.n_bins {
            let m = &mut mel[i * N_FRAMES + i_frame];
            let v = m.max(m_max);
            *m = (v + 4.0) / 4.0;
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
        self.mel.fill(MEL_ZERO);
        self.i_frame = 0;
    }
}
