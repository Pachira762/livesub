use crate::asr::{
    common::conv::{Conv1d, Conv1dConfig},
    common::tensor_ext::TensorExt,
    parakeet::attention::RelPositionMultiHeadAttention,
};
use candle::{D::Minus1, Module, ModuleT, Result, Tensor, conv::CudnnFwdAlgo};
use candle_nn::{
    Activation, BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, Linear, VarBuilder,
};

pub struct ConformerLayer {
    fc_factor: f64,
    norm_feed_forward1: LayerNorm,
    feed_forward1: ConformerFeedForward,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_self_att: LayerNorm,
    self_attn: RelPositionMultiHeadAttention,
    norm_feed_forward2: LayerNorm,
    feed_forward2: ConformerFeedForward,
    norm_out: LayerNorm,
}

impl ConformerLayer {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        conv_kernel_size: usize,
        conv_context_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc_factor = 0.5;
        let nom_congif = LayerNormConfig::default();
        let norm_feed_forward1 =
            candle_nn::layer_norm(d_model, nom_congif, vb.pp("norm_feed_forward1"))?;
        let feed_forward1 = ConformerFeedForward::new(d_model, d_ff, vb.pp("feed_forward1"))?;

        let norm_self_att = candle_nn::layer_norm(d_model, nom_congif, vb.pp("norm_self_att"))?;
        let self_attn = RelPositionMultiHeadAttention::new(n_heads, d_model, vb.pp("self_attn"))?;

        let norm_conv = candle_nn::layer_norm(d_model, nom_congif, vb.pp("norm_conv"))?;
        let conv =
            ConformerConvolution::new(d_model, conv_kernel_size, conv_context_size, vb.pp("conv"))?;

        let norm_feed_forward2 =
            candle_nn::layer_norm(d_model, nom_congif, vb.pp("norm_feed_forward2"))?;
        let feed_forward2 = ConformerFeedForward::new(d_model, d_ff, vb.pp("feed_forward2"))?;

        let norm_out = candle_nn::layer_norm(d_model, nom_congif, vb.pp("norm_out"))?;

        Ok(Self {
            fc_factor,
            norm_feed_forward1,
            feed_forward1,
            norm_conv,
            conv,
            norm_self_att,
            self_attn,
            norm_feed_forward2,
            feed_forward2,
            norm_out,
        })
    }

    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.norm_feed_forward1.forward(x)?;
        let x = self.feed_forward1.forward(&x)?;
        let residual = (residual + (self.fc_factor * x)?)?;

        let x = self.norm_self_att.forward(&residual)?;
        let x = self.self_attn.forward(&x, &x, &x, pos_emb)?;
        let residual = (residual + &x)?;

        let x = self.norm_conv.forward(&residual)?;
        let x = self.conv.forward(&x)?;
        let residual = (residual + &x)?;

        let x = self.norm_feed_forward2.forward(&residual)?;
        let x = self.feed_forward2.forward(&x)?;
        let residual = (residual + (self.fc_factor * x)?)?;

        let x = self.norm_out.forward(&residual)?;

        Ok(x)
    }
}

pub struct ConformerFeedForward {
    linear1: Linear,
    activation: Activation,
    linear2: Linear,
}

impl Module for ConformerFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.activation.forward(&x)?;
        let x = self.linear2.forward(&x)?;

        Ok(x)
    }
}

impl ConformerFeedForward {
    pub fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear_no_bias(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_no_bias(d_ff, d_model, vb.pp("linear2"))?;
        let activation = Activation::Silu;

        Ok(Self {
            linear1,
            activation,
            linear2,
        })
    }
}

pub struct ConformerConvolution {
    pointwise_conv1: Conv1d,
    pointwise_conv2: Conv1d,
    depthwise_conv: CausalConv1D,
    batch_norm: BatchNorm,
    activation: Activation,
}

impl ConformerConvolution {
    pub fn new(
        d_model: usize,
        kernel_size: usize,
        conv_context_size: (usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let dw_conv_input_dim = d_model;

        let config = Conv1dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };
        let pointwise_conv1 =
            Conv1d::new(d_model, 2 * d_model, 1, config, vb.pp("pointwise_conv1"))?;
        let pointwise_conv2 = Conv1d::new(
            dw_conv_input_dim,
            d_model,
            1,
            config,
            vb.pp("pointwise_conv2"),
        )?;

        let depthwise_conv = CausalConv1D::new(
            d_model,
            d_model,
            kernel_size,
            1,
            conv_context_size,
            d_model,
            vb.pp("depthwise_conv"),
        )?;

        let config = BatchNormConfig {
            eps: 1e-05,
            remove_mean: false,
            affine: true,
            momentum: 1.0,
        };
        let batch_norm = candle_nn::batch_norm(dw_conv_input_dim, config, vb.pp("batch_norm"))?;

        let activation = Activation::Silu;

        Ok(Self {
            pointwise_conv1,
            pointwise_conv2,
            depthwise_conv,
            batch_norm,
            activation,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?;
        let x = self.pointwise_conv1.forward(&x)?;
        let x = x.glu(1)?;
        let x = self.depthwise_conv.forward(&x)?;
        let x = self.batch_norm.forward_t(&x, false)?;
        let x = self.activation.forward(&x)?;
        let x = self.pointwise_conv2.forward(&x)?;
        let x = x.transpose(1, 2)?;

        Ok(x)
    }
}

struct CausalConv1D {
    left_padding: usize,
    right_padding: usize,
    conv: Conv1d,
}

impl CausalConv1D {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: (usize, usize),
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (padding, left_padding, right_padding) = if padding.0 == padding.1 {
            (padding.0, 0, 0)
        } else {
            (0, padding.0, padding.1)
        };

        let config = Conv1dConfig {
            padding,
            stride,
            groups,
            cudnn_fwd_algo: Some(CudnnFwdAlgo::Gemm),
            ..Default::default()
        };
        let conv = Conv1d::new(in_channels, out_channels, kernel_size, config, vb)?;

        Ok(Self {
            left_padding,
            right_padding,
            conv,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x: Tensor = if self.left_padding > 0 || self.right_padding > 0 {
            x.pad_with_zeros(Minus1, self.left_padding, self.right_padding)?
        } else {
            x.clone()
        };

        let x = self.conv.forward(&x)?;

        Ok(x)
    }
}
