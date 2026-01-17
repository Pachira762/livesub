use candle::{Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use crate::asr::common::conv::{Conv1d, Conv1dConfig};

pub struct Decoder {
    embedding: Embedding,
    conv: Conv1d,
    decoder_proj: Linear,
}

impl Decoder {
    pub fn new(
        n_vocab: usize,
        decoder_dim: usize,
        context_size: usize,
        joiner_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embedding = candle_nn::embedding(n_vocab, decoder_dim, vb.pp("embedding"))?;

        let cfg = Conv1dConfig {
            groups: decoder_dim / 4,
            ..Default::default()
        };
        let conv = Conv1d::new(decoder_dim, decoder_dim, context_size, cfg, vb.pp("conv"))?;

        let decoder_proj = candle_nn::linear(decoder_dim, joiner_dim, vb.pp("decoder_proj"))?;

        Ok(Self {
            embedding,
            conv,
            decoder_proj,
        })
    }

    pub fn forward(&self, y: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(y)?;
        let x = x.permute((0, 2, 1))?;
        let x = self.conv.forward(&x.contiguous()?)?;
        let x = x.permute((0, 2, 1))?;
        let x = x.relu()?;
        let x = self.decoder_proj.forward(&x)?;

        Ok(x)
    }
}
