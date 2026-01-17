use std::f32::consts::PI;

use candle::{D, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, ops::softmax};

use crate::asr::{
    common::conv::{Conv1d, Conv1dConfig},
    common::tensor_ext::TensorExt,
    reazonspeech::scaling::{BiasNorm, Scaling},
};

pub struct Zipformer2 {
    encoder_dim: Vec<usize>,
    encoders: Vec<Box<dyn Zipformer2EncoderModule>>,
    downsample_output: SimpleDownsample,
}

impl Zipformer2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output_downsampling_factor: usize,
        downsampling_factor: &[usize],
        encoder_dim: &[usize],
        num_encoder_layers: &[usize],
        query_head_dim: &[usize],
        pos_head_dim: &[usize],
        value_head_dim: &[usize],
        num_heads: &[usize],
        feedforward_dim: &[usize],
        cnn_module_kernel: &[usize],
        pos_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_encoders = downsampling_factor.len();
        let encoders: Vec<_> = (0..num_encoders)
            .map(|i| {
                let encoder: Box<dyn Zipformer2EncoderModule> = if downsampling_factor[i] == 1 {
                    let encoder = Zipformer2Encoder::new(
                        num_encoder_layers[i],
                        encoder_dim[i],
                        pos_dim,
                        num_heads[i],
                        query_head_dim[i],
                        pos_head_dim[i],
                        value_head_dim[i],
                        feedforward_dim[i],
                        cnn_module_kernel[i],
                        vb.pp(format!("encoders.{i}")),
                    )?;
                    Box::new(encoder)
                } else {
                    let encoder = Zipformer2Encoder::new(
                        num_encoder_layers[i],
                        encoder_dim[i],
                        pos_dim,
                        num_heads[i],
                        query_head_dim[i],
                        pos_head_dim[i],
                        value_head_dim[i],
                        feedforward_dim[i],
                        cnn_module_kernel[i],
                        vb.pp(format!("encoders.{i}.encoder")),
                    )?;
                    let encoder = DownsampledZipformer2Encoder::new(
                        encoder,
                        encoder_dim[i],
                        downsampling_factor[i],
                        vb.pp(format!("encoders.{i}")),
                    )?;
                    Box::new(encoder)
                };

                Ok(encoder)
            })
            .collect::<Result<_>>()?;

        let downsample_output =
            SimpleDownsample::new(output_downsampling_factor, vb.pp("downsample_output"))?;

        Ok(Self {
            encoder_dim: encoder_dim.into(),
            encoders,
            downsample_output,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = vec![];

        let mut x = x.clone();
        for i in 0..self.encoders.len() {
            x = Self::convert_num_channels(&x, self.encoder_dim[i])?;

            x = self.encoders[i].forward(&x)?;

            outputs.push(x.clone());
        }

        let x = self.full_dim_output(&outputs)?;
        let x = self.downsample_output.forward(&x)?;

        Ok(x)
    }

    fn convert_num_channels(x: &Tensor, num_channels: usize) -> Result<Tensor> {
        let (seq_len, batch_size, channels) = x.dims3()?;
        match num_channels.cmp(&channels) {
            std::cmp::Ordering::Less => x.i((.., .., ..num_channels)),
            std::cmp::Ordering::Equal => Ok(x.clone()),
            std::cmp::Ordering::Greater => {
                let pad = num_channels - channels;
                let zeros = Tensor::zeros((seq_len, batch_size, pad), x.dtype(), x.device())?;
                Tensor::cat(&[x, &zeros], 2)
            }
        }
    }

    fn full_dim_output(&self, outputs: &[Tensor]) -> Result<Tensor> {
        let num_encoders = self.encoder_dim.len();
        let mut output_pieces = vec![outputs.last().unwrap().clone()];

        let mut cur_dim = *self.encoder_dim.last().unwrap();
        for i in (0..=num_encoders - 2).rev() {
            let d = self.encoder_dim[i];
            if d > cur_dim {
                let this_output = &outputs[i];
                output_pieces.push(this_output.i((.., .., cur_dim..d))?);
                cur_dim = d;
            }
        }

        Tensor::cat(&output_pieces, D::Minus1)
    }

    #[allow(unused)]
    fn downsample_mask(
        &self,
        src_key_padding_mask: Option<&Tensor>,
        downsampling_factor: usize,
    ) -> Result<Option<Tensor>> {
        match &src_key_padding_mask {
            Some(mask) => {
                let len = mask.dim(D::Minus1)?;
                let indexes = Tensor::arange_step(
                    0u32,
                    len as u32,
                    downsampling_factor as u32,
                    mask.device(),
                )?;
                let mask = mask.index_select(&indexes, D::Minus1)?;
                Ok(Some(mask))
            }
            None => Ok(None),
        }
    }
}

#[derive(Clone, Debug)]
struct Zipformer2EncoderLayer {
    bypass: BypassModule,
    bypass_mid: BypassModule,
    self_attn_weights: RelPositionMultiheadAttentionWeights,
    self_attn1: SelfAttention,
    self_attn2: SelfAttention,
    feed_forward1: FeedforwardModule,
    feed_forward2: FeedforwardModule,
    feed_forward3: FeedforwardModule,
    nonlin_attention: NonlinAttention,
    conv_module1: ConvolutionModule,
    conv_module2: ConvolutionModule,
    norm: BiasNorm,
}

impl Zipformer2EncoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        embed_dim: usize,
        pos_dim: usize,
        num_heads: usize,
        query_head_dim: usize,
        pos_head_dim: usize,
        value_head_dim: usize,
        feedforward_dim: usize,
        cnn_module_kernel: usize,
        // causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let bypass = BypassModule::new(embed_dim, vb.pp("bypass"))?;
        let bypass_mid = BypassModule::new(embed_dim, vb.pp("bypass_mid"))?;

        let self_attn_weights = RelPositionMultiheadAttentionWeights::new(
            embed_dim,
            pos_dim,
            num_heads,
            query_head_dim,
            pos_head_dim,
            vb.pp("self_attn_weights"),
        )?;

        let self_attn1 =
            SelfAttention::new(embed_dim, num_heads, value_head_dim, vb.pp("self_attn1"))?;
        let self_attn2 =
            SelfAttention::new(embed_dim, num_heads, value_head_dim, vb.pp("self_attn2"))?;

        let feed_forward1 =
            FeedforwardModule::new(embed_dim, (feedforward_dim * 3) / 4, vb.pp("feed_forward1"))?;
        let feed_forward2 =
            FeedforwardModule::new(embed_dim, feedforward_dim, vb.pp("feed_forward2"))?;
        let feed_forward3 =
            FeedforwardModule::new(embed_dim, (feedforward_dim * 5) / 4, vb.pp("feed_forward3"))?;

        let nonlin_attention =
            NonlinAttention::new(embed_dim, 3 * embed_dim / 4, vb.pp("nonlin_attention"))?;

        let conv_module1 =
            ConvolutionModule::new(embed_dim, cnn_module_kernel, vb.pp("conv_module1"))?;
        let conv_module2 =
            ConvolutionModule::new(embed_dim, cnn_module_kernel, vb.pp("conv_module2"))?;

        let norm = BiasNorm::new(embed_dim, vb.pp("norm"))?;

        Ok(Self {
            bypass,
            bypass_mid,
            self_attn_weights,
            self_attn1,
            self_attn2,
            feed_forward1,
            feed_forward2,
            feed_forward3,
            nonlin_attention,
            conv_module1,
            conv_module2,
            norm,
        })
    }

    fn forward(&self, src: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let src_orig = src.clone();

        let attn_weights = self.self_attn_weights.forward(src, pos_emb)?;

        let ff = self.feed_forward1.forward(src)?;
        let src = (src + ff)?;

        let selected_attn_weights = attn_weights.i(0..1)?;
        let na = self
            .nonlin_attention
            .forward(&src, &selected_attn_weights)?;
        let src = (src + na)?;

        let self_attn = self.self_attn1.forward(&src, &attn_weights)?;
        let src = (src + &self_attn)?;

        let conv = self.conv_module1.forward(&src)?;
        let src = (src + &conv)?;

        let ff = self.feed_forward2.forward(&src)?;
        let src = (src + &ff)?;

        let src = self.bypass_mid.forward(&src_orig, &src)?;

        let self_attn = self.self_attn2.forward(&src, &attn_weights)?;
        let src = (src + &self_attn)?;

        let conv = self.conv_module2.forward(&src)?;
        let src = (src + &conv)?;

        let ff = self.feed_forward3.forward(&src)?;
        let src = (src + &ff)?;

        let src = self.norm.forward(&src)?;

        let src = self.bypass.forward(&src_orig, &src)?;

        Ok(src)
    }
}

trait Zipformer2EncoderModule {
    fn forward(&mut self, src: &Tensor) -> Result<Tensor>;
}

#[derive(Clone, Debug)]
struct Zipformer2Encoder {
    encoder_pos: CompactRelPositionalEncoding,
    layers: Vec<Zipformer2EncoderLayer>,
}

impl Zipformer2Encoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_layers: usize,
        embed_dim: usize,
        pos_dim: usize,
        num_heads: usize,
        query_head_dim: usize,
        pos_head_dim: usize,
        value_head_dim: usize,
        feedforward_dim: usize,
        cnn_module_kernel: usize,
        // causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let encoder_pos = CompactRelPositionalEncoding::new(pos_dim, 1.0)?;
        let layers: Vec<_> = (0..num_layers)
            .map(|i| {
                Zipformer2EncoderLayer::new(
                    embed_dim,
                    pos_dim,
                    num_heads,
                    query_head_dim,
                    pos_head_dim,
                    value_head_dim,
                    feedforward_dim,
                    cnn_module_kernel,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            encoder_pos,
            layers,
        })
    }
}

impl Zipformer2EncoderModule for Zipformer2Encoder {
    fn forward(&mut self, src: &Tensor) -> Result<Tensor> {
        let pos_emb = self.encoder_pos.forward(src)?;

        let mut output = src.clone();
        for layer in &self.layers {
            output = layer.forward(&output, &pos_emb)?;
        }

        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct DownsampledZipformer2Encoder {
    encoder: Zipformer2Encoder,
    downsample: SimpleDownsample,
    upsample: SimpleUpsample,
    out_combiner: BypassModule,
}

impl DownsampledZipformer2Encoder {
    fn new(
        encoder: Zipformer2Encoder,
        dim: usize,
        downsample_factor: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let downsample = SimpleDownsample::new(downsample_factor, vb.pp("downsample"))?;
        let upsample = SimpleUpsample::new(downsample_factor)?;
        let out_combiner = BypassModule::new(dim, vb.pp("out_combiner"))?;

        Ok(Self {
            encoder,
            downsample,
            upsample,
            out_combiner,
        })
    }
}

impl Zipformer2EncoderModule for DownsampledZipformer2Encoder {
    fn forward(&mut self, src: &Tensor) -> Result<Tensor> {
        let src_orig = src.clone();
        let src = self.downsample.forward(src)?;
        let src = self.encoder.forward(&src)?;
        let src = self.upsample.forward(&src)?;
        let src = src.i((..src_orig.size(0), .., ..))?;
        let out = self.out_combiner.forward(&src_orig, &src)?;

        Ok(out)
    }
}

#[derive(Clone, Debug)]
struct SimpleDownsample {
    downsampling_factor: usize,
    weight: Tensor,
}

impl SimpleDownsample {
    fn new(downsampling_factor: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((downsampling_factor, 1, 1), "weight")?;
        Ok(Self {
            downsampling_factor,
            weight,
        })
    }
}

impl Module for SimpleDownsample {
    fn forward(&self, src: &Tensor) -> Result<Tensor> {
        let (seq_len, batch_size, in_channels) = src.dims3()?;
        let ds = self.downsampling_factor;
        let d_seq_len = seq_len.div_ceil(ds);
        let pad = d_seq_len * ds - seq_len;

        let src_extra = src
            .i((seq_len - 1, .., ..))?
            .expand((pad, batch_size, in_channels))?;
        let src = Tensor::cat(&[src, &src_extra], 0)?;

        let src = src.reshape((d_seq_len, ds, batch_size, in_channels))?;

        let weight = self.weight.expand(src.shape())?;
        let ans = src.mul(&weight)?.sum(1)?;

        Ok(ans)
    }
}

#[derive(Clone, Debug)]
struct SimpleUpsample {
    upsample_factor: usize,
}

impl SimpleUpsample {
    fn new(upsample_factor: usize) -> Result<Self> {
        Ok(Self { upsample_factor })
    }
}

impl Module for SimpleUpsample {
    fn forward(&self, src: &Tensor) -> Result<Tensor> {
        let upsample = self.upsample_factor;
        let (seq_len, batch_size, num_channels) = src.dims3()?;
        let src = src
            .unsqueeze(1)?
            .expand((seq_len, upsample, batch_size, num_channels))?;
        let src = src.reshape((seq_len * upsample, batch_size, num_channels))?;

        Ok(src)
    }
}

#[derive(Clone, Debug)]
struct BypassModule {
    bypass_scale: Tensor,
}

impl BypassModule {
    fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let bypass_scale = vb.get(embed_dim, "bypass_scale")?;
        Ok(Self { bypass_scale })
    }

    fn forward(&self, src_orig: &Tensor, src: &Tensor) -> Result<Tensor> {
        src_orig + &((src - src_orig)?.broadcast_mul(&self.bypass_scale))?
    }
}

#[derive(Clone, Debug)]
struct CompactRelPositionalEncoding {
    embed_dim: usize,
    length_factor: f32,
    pe: Option<Tensor>,
}

impl CompactRelPositionalEncoding {
    fn new(embed_dim: usize, length_factor: f32) -> Result<Self> {
        Ok(Self {
            embed_dim,
            length_factor,
            pe: None,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let pe = self.extend_pe(x)?;

        let pe_len = pe.size(0);
        let beg = pe_len / 2 - x.size(0) + 1;
        let end = pe_len / 2 + x.size(0);
        let pos_emb = pe.i((beg..end, ..))?.unsqueeze(0)?;

        Ok(pos_emb)
    }

    fn extend_pe(&mut self, x: &Tensor) -> Result<Tensor> {
        let x_len = x.size(0).max(2); // avoid 0 len

        if let Some(pe) = &self.pe
            && pe.size(0) >= 2 * x_len - 1
        {
            return Ok(pe.clone());
        }

        let pe = Self::generate_pe(x_len, self.embed_dim, self.length_factor, x.device())?;
        self.pe = Some(pe.clone());

        Ok(pe)
    }

    fn generate_pe(
        x_len: usize,
        n_dims: usize,
        length_factor: f32,
        device: &Device,
    ) -> Result<Tensor> {
        let x_len = x_len as i64;
        let compression_length = (n_dims as f32).sqrt();
        let log_compression_length = compression_length.ln();
        let length_scale = length_factor * n_dims as f32 / (2.0 * PI);

        let x: Vec<_> = (-x_len..x_len)
            .map(|i| {
                let x = i as f32;

                let x = compression_length
                    * x.signum()
                    * ((x.abs() + compression_length).ln() - log_compression_length);

                (x / length_scale).atan()
            })
            .collect();

        let freqs: Vec<_> = (0..n_dims / 2).map(|i| (i + 1) as f32).collect();

        let pe_len = x.len();
        let mut pe = vec![0.0f32; pe_len * n_dims];
        for i in 0..pe_len {
            for j in 0..n_dims / 2 {
                let t = freqs[j] * x[i];
                pe[i * n_dims + 2 * j] = t.cos();
                pe[i * n_dims + 2 * j + 1] = t.sin();
            }
            pe[i * n_dims + n_dims - 1] = 1.0; // for bias
        }

        Tensor::new(pe, device)?.reshape((pe_len, n_dims))
    }
}

#[derive(Clone, Debug)]
struct RelPositionMultiheadAttentionWeights {
    query_head_dim: usize,
    pos_head_dim: usize,
    num_heads: usize,
    in_proj: Linear,
    linear_pos: Linear,
}

impl RelPositionMultiheadAttentionWeights {
    fn new(
        embed_dim: usize,
        pos_dim: usize,
        num_heads: usize,
        query_head_dim: usize,
        pos_head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let key_head_dim = query_head_dim;
        let in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads;

        let in_proj = candle_nn::linear(embed_dim, in_proj_dim, vb.pp("in_proj"))?;
        let linear_pos =
            candle_nn::linear_no_bias(pos_dim, num_heads * pos_head_dim, vb.pp("linear_pos"))?;

        Ok(Self {
            query_head_dim,
            pos_head_dim,
            num_heads,
            in_proj,
            linear_pos,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;

        let query_head_dim = self.query_head_dim;
        let pos_head_dim = self.pos_head_dim;
        let num_heads = self.num_heads;
        let (seq_len, batch_size, _) = x.dims3()?;
        let query_dim = query_head_dim * num_heads;

        let q = x.i((.., .., ..query_dim))?;
        let k = x.i((.., .., query_dim..2 * query_dim))?;
        let p = x.i((.., .., 2 * query_dim..))?;
        assert_eq!(p.dim(candle::D::Minus1)?, num_heads * pos_head_dim);

        let q = q.reshape((seq_len, batch_size, num_heads, query_head_dim))?;
        let p = p.reshape((seq_len, batch_size, num_heads, pos_head_dim))?;
        let k = k.reshape((seq_len, batch_size, num_heads, query_head_dim))?;

        let q = q.permute((2, 1, 0, 3))?.contiguous()?;
        let p = p.permute((2, 1, 0, 3))?.contiguous()?;
        let k = k.permute((2, 1, 3, 0))?.contiguous()?;

        let attn_scores = q.matmul(&k)?;

        let pos_emb = self.linear_pos.forward(pos_emb)?;
        let seq_len2 = 2 * seq_len - 1;
        let pos_emb = pos_emb
            .reshape(((), seq_len2, num_heads, pos_head_dim))?
            .permute((2, 0, 3, 1))?;
        let pos_scores = p.matmul(&pos_emb.contiguous()?)?;

        let (num_heads, batch_size, time1, n) = pos_scores.dims4()?;
        let rows = Tensor::arange_step(time1 as i64 - 1, -1, -1, x.device())?;
        let cols = Tensor::arange(0, seq_len as i64, x.device())?;
        let rows = rows.repeat(batch_size * num_heads)?.unsqueeze(D::Minus1)?;
        let indexes = rows.broadcast_add(&cols)?;
        let pos_scores = pos_scores.reshape(((), n))?;
        let pos_scores = pos_scores.gather(&indexes, 1)?;
        let pos_scores = pos_scores.reshape((num_heads, batch_size, time1, seq_len))?;

        let attn_scores = (attn_scores + pos_scores)?;
        let attn_weights = softmax(&attn_scores, D::Minus1)?.contiguous()?;

        Ok(attn_weights)
    }
}

#[derive(Clone, Debug)]
struct SelfAttention {
    in_proj: Linear,
    out_proj: Linear,
}

impl SelfAttention {
    fn new(
        embed_dim: usize,
        num_heads: usize,
        value_head_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_proj = candle_nn::linear(embed_dim, num_heads * value_head_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear(num_heads * value_head_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self { in_proj, out_proj })
    }

    fn forward(&self, x: &Tensor, attn_weights: &Tensor) -> Result<Tensor> {
        let (seq_len, batch_size, _embed_dim) = x.dims3()?;
        let num_heads = attn_weights.size(0);
        assert_eq!(
            attn_weights.shape(),
            &candle::Shape::from((num_heads, batch_size, seq_len, seq_len))
        );

        let x = self.in_proj.forward(x)?;
        let x = x
            .reshape((seq_len, batch_size, num_heads, ()))?
            .permute((2, 1, 0, 3))?;
        let value_head_dim = x.size(3);

        let x = attn_weights.matmul(&x.contiguous()?)?;

        let x =
            x.permute((2, 1, 0, 3))?
                .reshape((seq_len, batch_size, num_heads * value_head_dim))?;

        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}

#[derive(Clone, Debug)]
struct FeedforwardModule {
    in_proj: Linear,
    out_proj: Linear,
}

impl FeedforwardModule {
    fn new(embed_dim: usize, feedforward_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj = candle_nn::linear(embed_dim, feedforward_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear(feedforward_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self { in_proj, out_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;
        let x = x.swoosh_l()?;
        let x = self.out_proj.forward(&x)?;
        Ok(x)
    }
}

#[derive(Clone, Debug)]
struct NonlinAttention {
    hidden_channels: usize,
    in_proj: Linear,
    out_proj: Linear,
}

impl NonlinAttention {
    fn new(channels: usize, hidden_channels: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj = candle_nn::linear(channels, hidden_channels * 3, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear(hidden_channels, channels, vb.pp("out_proj"))?;
        Ok(Self {
            hidden_channels,
            in_proj,
            out_proj,
        })
    }

    fn forward(&self, x: &Tensor, attn_weights: &Tensor) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;

        let (seq_len, batch_size, _) = x.dims3()?;
        let hidden_channels = self.hidden_channels;

        let (s, x, y) = {
            let chunks = x.chunk(3, 2)?;
            (chunks[0].clone(), chunks[1].clone(), chunks[2].clone())
        };

        let s = s.tanh()?;

        let s = s
            .unsqueeze(D::Minus1)?
            .reshape((seq_len, batch_size, hidden_channels))?;
        let x = (x * s)?;

        let (seq_len, batch_size, _embed_dim) = x.dims3()?;
        let num_heads = attn_weights.size(0);
        assert_eq!(
            attn_weights.shape(),
            &candle::Shape::from((num_heads, batch_size, seq_len, seq_len))
        );

        let x = x
            .reshape((seq_len, batch_size, num_heads, ()))?
            .permute((2, 1, 0, 3))?;
        let x = attn_weights.matmul(&x.contiguous()?)?;
        let x = x
            .permute((2, 1, 0, 3))?
            .reshape((seq_len, batch_size, ()))?;

        let x = (x * y)?;

        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}

#[derive(Clone, Debug)]
struct ConvolutionModule {
    in_proj: Linear,
    depthwise_conv: Conv1d,
    out_proj: Linear,
}

impl ConvolutionModule {
    fn new(channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        assert_eq!((kernel_size - 1) % 2, 0);

        let bottleneck_dim = channels;

        let in_proj = candle_nn::linear(channels, 2 * bottleneck_dim, vb.pp("in_proj"))?;

        let depthwise_conv = Conv1d::new(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size,
            Conv1dConfig {
                padding: kernel_size / 2,
                groups: bottleneck_dim,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let out_proj = candle_nn::linear(bottleneck_dim, channels, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            depthwise_conv,
            out_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;

        let (x, s) = {
            let x = x.chunk(2, 2)?;
            (x[0].clone(), x[1].clone())
        };
        let s = candle_nn::ops::sigmoid(&s)?;
        let x = (x * s)?;
        let x = x.permute((1, 2, 0))?;

        let x = self.depthwise_conv.forward(&x)?;
        let x = x.permute((2, 0, 1))?;

        let x = x.swoosh_r()?;
        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}
