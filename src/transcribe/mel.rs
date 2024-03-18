use anyhow::Result;
use candle_transformers::models::whisper::{self as m, HOP_LENGTH as N_HOP, N_FFT};
use rustfft::{num_complex::Complex32 as Complex, Fft, FftPlanner};
use std::{f32::consts::PI, sync::Arc};

use super::common::Shiftable;

const N_FILTER: usize = (N_FFT / 2) + 1;
const N_FRAMES: usize = m::N_FRAMES;
const N_BUF: usize = N_FRAMES / 30;
const MEL_ZERO: f32 = (-10.0 + 4.0) / 4.0;

pub struct MelDecodeResult {
    pub n_frames: usize,
    pub new_segment: bool,
}

pub struct MelSpectrogram {
    n_mel: usize,
    mel: Vec<f32>,
    i_frame: usize,
    samples: Vec<f32>,
    decoder: MelDecoder,
}

impl MelSpectrogram {
    pub fn new(num_mel_bins: usize) -> Result<Self> {
        Ok(Self {
            n_mel: num_mel_bins,
            mel: vec![MEL_ZERO; num_mel_bins * N_FRAMES],
            i_frame: 0,
            samples: vec![],
            decoder: MelDecoder::new(num_mel_bins)?,
        })
    }

    pub fn process(&mut self, pcm: &[f32]) -> MelDecodeResult {
        self.samples.extend_from_slice(pcm);

        if self.samples.len() < N_FFT {
            return MelDecodeResult {
                n_frames: 0,
                new_segment: false,
            };
        }

        let n_samples = self.samples.len();
        let n_frames = num_frames(n_samples);
        let new_segment = if self.i_frame + n_frames < N_FRAMES {
            false
        } else {
            self.shift();
            true
        };

        let n_frames = n_frames.min(N_FRAMES - self.i_frame);
        self.pcm_to_mel(n_frames);

        let n_samples = num_samples(n_frames);
        self.samples.shift(n_samples - (N_FFT - N_HOP));

        MelDecodeResult {
            n_frames,
            new_segment,
        }
    }

    pub fn clear(&mut self) {
        self.mel.fill(MEL_ZERO);
        self.i_frame = 0;
        self.samples.clear();
    }

    pub fn mel(&self) -> &[f32] {
        &self.mel
    }

    fn shift(&mut self) {
        let n_copy = if self.i_frame < N_BUF {
            self.i_frame
        } else {
            N_BUF
        };

        for i in 0..self.n_mel {
            let offset = N_FRAMES * i;
            let beg = offset + self.i_frame - n_copy;
            let end = beg + n_copy;
            self.mel.copy_within(beg..end, offset);

            let beg = offset + n_copy;
            let end = offset + N_FRAMES;
            self.mel[beg..end].fill(MEL_ZERO);
        }

        self.i_frame = n_copy;
    }

    fn pcm_to_mel(&mut self, n_frames: usize) {
        self.decoder
            .process(&self.samples, n_frames, self.i_frame, &mut self.mel);

        self.i_frame += n_frames;
    }
}

struct MelDecoder {
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    fft_io: Vec<Complex>,
    fft_scratch: Vec<Complex>,
    magnitude: Vec<f32>,
    filter: Vec<f32>,
    n_mel: usize,
}

impl MelDecoder {
    fn new(num_mel_bins: usize) -> Result<Self> {
        let window = (0..N_FFT)
            .map(|i| 0.5 * (1. - ((2.0 * PI * i as f32) / (N_FFT - 1) as f32).cos()))
            .collect();

        let fft = FftPlanner::new().plan_fft_forward(N_FFT);
        let n_scratch = fft.get_inplace_scratch_len();

        let filter = {
            let mel_bytes = match num_mel_bins {
                80 => include_bytes!("melfilters.bytes").as_slice(),
                128 => include_bytes!("melfilters128.bytes").as_slice(),
                nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
            };

            let mut filter = vec![0f32; mel_bytes.len() / 4];
            <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
                mel_bytes,
                &mut filter,
            );

            filter
        };

        Ok(Self {
            window,
            fft,
            fft_io: vec![Complex::default(); N_FFT],
            fft_scratch: vec![Complex::default(); n_scratch],
            magnitude: vec![0f32; N_FILTER],
            filter,
            n_mel: num_mel_bins,
        })
    }

    fn process(&mut self, samples: &[f32], n_frames: usize, i_frame: usize, mel: &mut [f32]) {
        let n_mel = self.n_mel;
        let window = self.window.as_slice();
        let fft = &mut self.fft;
        let io = &mut self.fft_io.as_mut_slice();
        let scratch = &mut self.fft_scratch.as_mut_slice();
        let magnitude = &mut self.magnitude.as_mut_slice();
        let filter = self.filter.as_slice();

        let mut mmax = 0f32;
        for i in 0..n_frames {
            let offset = N_HOP * i;
            for j in 0..N_FFT {
                io[j].re = window[j] * samples[offset + j];
                io[j].im = 0.;
            }

            fft.process_with_scratch(io, scratch);

            for j in 0..N_FILTER {
                magnitude[j] = io[j].norm_sqr();
            }

            for j in 0..n_mel {
                let mut sum = 0.;

                for k in 0..N_FILTER {
                    sum += filter[j * N_FILTER + k] * magnitude[k];
                }

                let m = sum.max(1e-10).log10();
                mel[j * N_FRAMES + i_frame + i] = m;
                mmax = mmax.max(m);
            }
        }
        mmax -= 8.0;

        for i in 0..n_frames {
            for j in 0..n_mel {
                let m = &mut mel[j * N_FRAMES + i_frame + i];
                let v = m.max(mmax);
                *m = (v + 4.0) / 4.0;
            }
        }
    }
}

fn num_frames(n_samples: usize) -> usize {
    if n_samples < N_FFT {
        0
    } else {
        (n_samples - (N_FFT - N_HOP)) / N_HOP
    }
}

fn num_samples(n_frames: usize) -> usize {
    if n_frames == 0 {
        0
    } else {
        (n_frames - 1) * N_HOP + N_FFT
    }
}
