use crate::asr::common::conv::{Conv2d, Conv2dConfig};
use candle::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub struct ConvSubsampling {
    conv_0: Conv2d,
    conv_2: Conv2d,
    conv_3: Conv2d,
    conv_5: Conv2d,
    conv_6: Conv2d,
    out: Linear,
}

impl ConvSubsampling {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let conv_0 = Conv2d::new(
            1,
            256,
            (3, 3),
            Conv2dConfig {
                padding: (1, 1),
                stride: (2, 2),
                ..Default::default()
            },
            vb.pp("conv.0"),
        )?;
        let conv_2 = Conv2d::new(
            256,
            256,
            (3, 3),
            Conv2dConfig {
                padding: (1, 1),
                stride: (2, 2),
                groups: 256,
                ..Default::default()
            },
            vb.pp("conv.2"),
        )?;
        let conv_3 = Conv2d::new(
            256,
            256,
            (1, 1),
            Conv2dConfig {
                ..Default::default()
            },
            vb.pp("conv.3"),
        )?;
        let conv_5 = Conv2d::new(
            256,
            256,
            (3, 3),
            Conv2dConfig {
                padding: (1, 1),
                stride: (2, 2),
                groups: 256,
                ..Default::default()
            },
            vb.pp("conv.5"),
        )?;
        let conv_6 = Conv2d::new(
            256,
            256,
            (1, 1),
            Conv2dConfig {
                ..Default::default()
            },
            vb.pp("conv.6"),
        )?;

        let out = candle_nn::linear(4096, 1024, vb.pp("out"))?;

        Ok(Self {
            conv_0,
            conv_2,
            conv_3,
            conv_5,
            conv_6,
            out,
        })
    }

    /// [B, T, F] => [B, (T - 7) / 8, H]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv(x)?;
        let x = self.proj(&x)?;

        Ok(x)
    }

    /// input shape (batch, seq_len, feat_dim)
    /// output shape (batch, ch, (seq_len - 7) / 8, 4096)
    pub fn conv(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.unsqueeze(1)?;

        x = self.conv_0.forward(&x)?;
        x = x.relu()?;
        x = self.conv_2.forward(&x)?;
        x = self.conv_3.forward(&x)?;
        x = x.relu()?;
        x = self.conv_5.forward(&x)?;
        x = self.conv_6.forward(&x)?;
        x = x.relu()?;

        Ok(x)
    }

    // [B, C, T, F] => [B, T, H]
    pub fn proj(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, t, _f) = x.dims4()?;
        let x = x.transpose(1, 2)?.reshape((b, t, ()))?;
        let x = self.out.forward(&x)?;

        Ok(x)
    }
}
