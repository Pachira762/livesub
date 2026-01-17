use candle::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, LayerNormConfig, VarBuilder};

use crate::asr::whisper::{attention::ResidualAttentionBlock, config::Config};

pub struct TextDecoder {
    embed_tokens: Embedding,
    embed_positions: Tensor,
    layers: Vec<ResidualAttentionBlock>,
    layer_norm: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.d_model, vb.pp("embed_tokens"))?;
        let embed_positions = vb.get(
            (config.max_target_positions, config.d_model),
            "embed_positions.weight",
        )?;

        let layers: Vec<_> = (0..config.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::new(
                    config.d_model,
                    config.decoder_attention_heads,
                    config.decoder_ffn_dim,
                    true,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<_>>()?;

        let layer_norm = candle_nn::layer_norm(
            config.d_model,
            LayerNormConfig::default(),
            vb.pp("layer_norm"),
        )?;

        let mask: Vec<_> = (0..config.d_model)
            .flat_map(|i| {
                (0..config.d_model).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (config.d_model, config.d_model), vb.device())?;

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            mask,
        })
    }

    pub fn forward(&mut self, x: &Tensor, xa: &Tensor, flush_kv_cache: bool) -> Result<Tensor> {
        let (_batch, len) = x.dims2()?;

        let mut x = self
            .embed_tokens
            .forward(&x.to_dtype(DType::U32)?)?
            .broadcast_add(&self.embed_positions.i(..len)?)?;

        for layer in &mut self.layers {
            x = layer.forward(&x, Some(xa), Some(&self.mask), flush_kv_cache)?;
        }
        x = self.layer_norm.forward(&x)?;

        let (_, seq_len, _) = x.dims3()?;
        let w = self.embed_tokens.embeddings().unsqueeze(0)?;
        x = x.i((.., seq_len - 1.., ..))?.matmul(&w.t()?)?;

        Ok(x)
    }
}
