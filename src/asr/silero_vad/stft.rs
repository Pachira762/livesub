use std::sync::Arc;

use anyhow::Result;
use candle::{Device, Tensor};
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};

use crate::asr::common::window;

pub struct Stft {
    device: Device,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
    buf: Vec<f32>,
    context: Vec<f32>,
}

impl Stft {
    const N_FFT: usize = 129;
    const N_FRAMES: usize = 4;

    pub fn new(sample_rate: u32, device: Device) -> Result<Self> {
        let chunk_len = 512 * sample_rate as usize / 16000;
        let window_len = chunk_len / 2;
        let window = window::hanning(window_len);

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(window_len);
        let fft_in = fft.make_input_vec();
        let fft_out = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();

        let buf = vec![0.0; Self::N_FFT * Self::N_FRAMES];
        let context = vec![0.0; chunk_len / 8];

        Ok(Self {
            device,
            window,
            fft,
            fft_in,
            fft_out,
            fft_scratch,
            buf,
            context,
        })
    }

    pub fn process(&mut self, samples: &[f32]) -> Result<Tensor> {
        let hop_len = self.window.len() / 2;
        let ctx_len = self.context.len();

        for i in 0..Self::N_FRAMES {
            for j in 0..self.window.len() {
                let idx = i * hop_len + j;
                if idx < ctx_len {
                    self.fft_in[j] = self.window[j] * self.context[idx];
                } else if idx < ctx_len + samples.len() {
                    self.fft_in[j] = self.window[j] * samples[idx - ctx_len];
                } else {
                    let idx = samples.len() - (idx - ctx_len - samples.len()) - 1;
                    self.fft_in[j] = self.window[j] * samples[idx];
                }
            }

            self.fft
                .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch)
                .unwrap();

            for j in 0..Self::N_FFT {
                self.buf[j * Self::N_FRAMES + i] = self.fft_out[j].norm();
            }
        }

        let context = &samples[samples.len() - self.context.len()..];
        self.context.copy_from_slice(context);

        let x = Tensor::from_slice(&self.buf, (1, Self::N_FFT, Self::N_FRAMES), &self.device)?;
        Ok(x)
    }

    pub fn reset(&mut self) {
        self.context.fill(0.0);
    }

    pub fn window_len(&self) -> usize {
        self.window.len()
    }
}
