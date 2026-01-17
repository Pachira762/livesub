use candle::{D, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::asr::common::tensor_ext::TensorExt;

#[derive(Clone, Debug)]
pub struct BiasNorm {
    bias: Tensor,
    log_scale_exp: Tensor,
}

impl BiasNorm {
    pub fn new(n_channels: usize, vb: VarBuilder) -> Result<Self> {
        let bias = vb.get(n_channels, "bias")?;
        let log_scale_exp = vb.get(1, "log_scale")?.exp()?;
        Ok(Self {
            bias,
            log_scale_exp,
        })
    }
}

impl Module for BiasNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let scale = x
            .broadcast_sub(&self.bias)?
            .powf(2.0)?
            .mean_keepdim(D::Minus1)?
            .powf(-0.5)?
            .broadcast_mul(&self.log_scale_exp)?;
        x.broadcast_mul(&scale)
    }
}

pub trait Scaling {
    fn swoosh_l(&self) -> Result<Tensor>;
    fn swoosh_r(&self) -> Result<Tensor>;
}

impl Scaling for Tensor {
    fn swoosh_l(&self) -> Result<Tensor> {
        self.scalar_sub(4.0)?
            .exp()?
            .scalar_add(1.0)?
            .log()?
            .sub(&self.scalar_mul(0.08)?)?
            .scalar_sub(0.035)
    }

    fn swoosh_r(&self) -> Result<Tensor> {
        self.scalar_sub(1.0)?
            .exp()?
            .scalar_add(1.0)?
            .log()?
            .sub(&self.scalar_mul(0.08)?)?
            .scalar_sub(0.313_261_7)
    }
}

#[derive(Clone, Debug)]
pub struct SwooshL {}

impl SwooshL {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for SwooshL {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.swoosh_l()
    }
}

#[derive(Clone, Debug)]
pub struct SwooshR {}

impl SwooshR {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for SwooshR {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.swoosh_r()
    }
}
