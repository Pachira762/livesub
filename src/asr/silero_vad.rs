use anyhow::Result;
use candle::{IndexOp, Module};
use candle_nn::VarBuilder;

use self::{decoder::Decoder, encoder::Encoder, stft::Stft};

mod decoder;
mod encoder;
mod stft;

pub struct SileroVad {
    stft: Stft,
    encoder: Encoder,
    decoder: Decoder,
}

impl SileroVad {
    pub fn new(sample_rate: u32, vb: VarBuilder) -> Result<Self> {
        let stft = Stft::new(sample_rate, vb.device().clone())?;

        let encoder = Encoder::new(
            &[129, 128, 64, 64],
            &[128, 64, 64, 128],
            &[3, 3, 3, 3],
            &[1, 2, 2, 1],
            &[1, 1, 1, 1],
            vb.pp("encoder"),
        )?;
        let decoder = Decoder::new(vb.pp("decoder"))?;

        Ok(Self {
            stft,
            encoder,
            decoder,
        })
    }

    pub fn infer(&mut self, samples: &[f32]) -> Result<f32> {
        let xs = self.stft.process(samples)?;
        let xs = self.encoder.forward(&xs)?;
        let xs = self.decoder.forward(&xs)?;
        let prob = xs.i((0, 0, 0))?.to_scalar()?;
        Ok(prob)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.stft.reset();
        self.decoder.reset()?;
        Ok(())
    }

    pub fn chunk_len(&self) -> usize {
        2 * self.stft.window_len()
    }
}
