use candle::{DType, Result, Tensor, shape::Dim};

pub trait TensorExt {
    fn size(&self, dim: usize) -> usize;
    fn float(&self) -> Result<Tensor>;
    fn scalar_add(&self, value: f64) -> Result<Tensor>;
    fn scalar_sub(&self, value: f64) -> Result<Tensor>;
    fn scalar_mul(&self, value: f64) -> Result<Tensor>;
    fn scalar_div(&self, value: f64) -> Result<Tensor>;
    fn glu<D: Dim>(&self, dim: D) -> Result<Tensor>;
    fn softmax<D: Dim>(&self, dim: D) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn size(&self, dim: usize) -> usize {
        self.dims()[dim]
    }

    fn float(&self) -> Result<Tensor> {
        self.to_dtype(DType::F32)
    }

    fn scalar_add(&self, value: f64) -> Result<Tensor> {
        self + value
    }

    fn scalar_sub(&self, value: f64) -> Result<Tensor> {
        self - value
    }

    fn scalar_mul(&self, value: f64) -> Result<Tensor> {
        self * value
    }

    fn scalar_div(&self, value: f64) -> Result<Tensor> {
        self / value
    }

    fn glu<D: Dim>(&self, dim: D) -> Result<Tensor> {
        let chunks = self.chunk(2, dim)?;
        let a = &chunks[0];
        let b = &chunks[1];
        a.mul(&candle_nn::ops::sigmoid(b)?)
    }

    fn softmax<D: Dim>(&self, dim: D) -> Result<Tensor> {
        candle_nn::ops::softmax(self, dim)
    }
}
