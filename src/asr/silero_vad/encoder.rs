use anyhow::Result;
use candle::Module;
use candle_nn::{Sequential, VarBuilder};

use crate::asr::common::conv::{Conv1d, Conv1dConfig};

pub struct Encoder {
    layers: Sequential,
}

impl Encoder {
    pub fn new(
        in_channels: &[usize],
        out_channels: &[usize],
        kernel_sizes: &[usize],
        strides: &[usize],
        paddings: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = candle_nn::seq();

        for i in 0..in_channels.len() {
            layers = layers.add(EncoderLayer::new(
                in_channels[i],
                out_channels[i],
                kernel_sizes[i],
                strides[i],
                paddings[i],
                vb.pp(format!("{i}")),
            )?);
        }

        Ok(Self { layers })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &candle::Tensor) -> candle::Result<candle::Tensor> {
        self.layers.forward(xs)
    }
}

struct EncoderLayer {
    reparam_conv: Conv1d,
}

impl EncoderLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding,
            stride,
            ..Default::default()
        };
        let reparam_conv = Conv1d::new(
            in_channels,
            out_channels,
            kernel_size,
            cfg,
            vb.pp("reparam_conv"),
        )?;

        Ok(Self { reparam_conv })
    }
}

impl Module for EncoderLayer {
    fn forward(&self, xs: &candle::Tensor) -> candle::Result<candle::Tensor> {
        let xs = self.reparam_conv.forward(xs)?;
        let xs = xs.relu()?;
        Ok(xs)
    }
}
