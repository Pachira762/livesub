use candle::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

pub struct Joiner {
    output_linear: Linear,
}

impl Joiner {
    pub fn new(joiner_dim: usize, vocab_size: usize, vb: VarBuilder) -> Result<Self> {
        let output_linear = candle_nn::linear(joiner_dim, vocab_size, vb.pp("output_linear"))?;

        Ok(Self { output_linear })
    }

    pub fn forward(&self, encoder_out: &Tensor, decoder_out: &Tensor) -> Result<Tensor> {
        let x = (encoder_out + decoder_out)?;
        let x = x.tanh()?;
        let x = self.output_linear.forward(&x)?;

        Ok(x)
    }
}
