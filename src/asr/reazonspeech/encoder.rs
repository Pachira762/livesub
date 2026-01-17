use crate::asr::{
    common::tensor_ext::TensorExt,
    reazonspeech::{subsampling::Conv2dSubsampling, zipformer::Zipformer2},
};
use candle::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

pub struct Encoder {
    encoder_embed: Conv2dSubsampling,
    encoder: Zipformer2,
    encoder_proj: Linear,
}

impl Encoder {
    pub fn new(joiner_dim: usize, vb: VarBuilder) -> Result<Self> {
        let feature_dim = 80;
        let output_downsampling_factor = 2;
        let downsampling_factor = [1, 2, 4, 8, 4, 2];
        let num_encoder_layers = [2, 2, 4, 5, 4, 2];
        let feedforward_dim = [512, 768, 1536, 2048, 1536, 768];
        let encoder_dim = [192, 256, 512, 768, 512, 256];
        let num_heads = [4, 4, 4, 8, 4, 4];
        let query_head_dim = [32, 32, 32, 32, 32, 32];
        let value_head_dim = [12, 12, 12, 12, 12, 12];
        let pos_head_dim = [4, 4, 4, 4, 4, 4];
        let pos_dim = 48;
        let cnn_module_kernel = [31, 31, 15, 15, 15, 31];

        let encoder_embed =
            Conv2dSubsampling::new(feature_dim, encoder_dim[0], vb.pp("encoder_embed"))?;

        let encoder = Zipformer2::new(
            output_downsampling_factor,
            &downsampling_factor,
            &encoder_dim,
            &num_encoder_layers,
            &query_head_dim,
            &pos_head_dim,
            &value_head_dim,
            &num_heads,
            &feedforward_dim,
            &cnn_module_kernel,
            pos_dim,
            vb.pp("encoder"),
        )?;

        let encoder_proj = candle_nn::linear(
            *feedforward_dim.last().unwrap(),
            joiner_dim,
            vb.pp("encoder_proj"),
        )?;

        Ok(Self {
            encoder_embed,
            encoder,
            encoder_proj,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Option<Tensor>> {
        const MIN_INPUT_LENGTH: usize = 21;

        if x.size(1) < MIN_INPUT_LENGTH {
            return Ok(None);
        }

        let x = self.encoder_embed.forward(x)?;
        let x = x.permute((1, 0, 2))?;
        let x = self.encoder.forward(&x)?;

        let x = x.permute((1, 0, 2))?;
        let x = self.encoder_proj.forward(&x.contiguous()?)?;

        Ok(Some(x))
    }

    pub fn clear(&mut self) -> Result<()> {
        Ok(())
    }
}
