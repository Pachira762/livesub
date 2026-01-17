use candle::{Device, IndexOp, Module, Result, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, VarBuilder};

use crate::asr::{
    common::{
        conv::{Conv1d, Conv1dConfig},
        tensor_ext::TensorExt,
    },
    whisper::{attention::ResidualAttentionBlock, config::Config},
};

pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    layers: Vec<ResidualAttentionBlock>,
    layer_norm: LayerNorm,
}

impl AudioEncoder {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let conv1 = Conv1d::new(
            config.num_mel_bins,
            config.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = Conv1d::new(
            config.d_model,
            config.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        let positional_embedding = sigmoids(
            config.max_source_positions,
            config.d_model,
            10000,
            vb.device(),
        )?;

        let layers: Vec<_> = (0..config.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::new(
                    config.d_model,
                    config.encoder_attention_heads,
                    config.encoder_ffn_dim,
                    false,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<_>>()?;

        let layer_norm = candle_nn::layer_norm(
            config.d_model,
            LayerNormConfig::default(),
            vb.pp("layer_norm"),
        )?;

        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            layers,
            layer_norm,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;

        let (_b, t, _h) = x.dims3()?;
        let positional_embedding = self.positional_embedding.i((0..t, ..))?;
        let mut x = x.broadcast_add(&positional_embedding)?;

        for layer in &mut self.layers {
            x = layer.forward(&x, None, None, true)?;
        }

        let x = self.layer_norm.forward(&x)?;

        Ok(x)
    }
}

fn sigmoids(
    length: usize,
    channels: usize,
    max_timescale: usize,
    device: &Device,
) -> Result<Tensor> {
    let log_timescale_increment = (max_timescale as f64).ln() / ((channels / 2) as f64 - 1.0);
    let inv_timescales = Tensor::arange(0, channels as i64 / 2, device)?
        .float()?
        .scalar_mul(-log_timescale_increment)?
        .exp()?;
    let scaled_time = Tensor::arange(0, length as i64, device)?
        .float()?
        .unsqueeze(1)?
        .broadcast_mul(&inv_timescales.unsqueeze(0)?)?;
    Tensor::cat(&[Tensor::sin(&scaled_time)?, Tensor::cos(&scaled_time)?], 1)
}
