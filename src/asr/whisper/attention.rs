use candle::{D, IndexOp, Module, Result, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, VarBuilder};

use crate::asr::common::tensor_ext::TensorExt;

pub struct MultiHeadAttention {
    n_head: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl MultiHeadAttention {
    pub fn new(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let q_proj = candle_nn::linear(n_state, n_state, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(n_state, n_state, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(n_state, n_state, vb.pp("out_proj"))?;

        Ok(Self {
            n_head,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        if flush_kv_cache {
            self.kv_cache = None;
        }

        let q = self.q_proj.forward(x)?;
        let (k, v) = match xa {
            Some(xa) => match &self.kv_cache {
                Some((k, v)) => (k.clone(), v.clone()),
                None => {
                    let k = self.k_proj.forward(xa)?;
                    let v = self.v_proj.forward(xa)?;
                    self.kv_cache = Some((k.clone(), v.clone()));
                    (k, v)
                }
            },
            None => (self.k_proj.forward(x)?, self.v_proj.forward(x)?),
        };

        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let x = self.out_proj.forward(&wv)?;

        Ok(x)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = q
            .reshape((n_batch, q.size(1), self.n_head, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((n_batch, k.size(1), self.n_head, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((n_batch, v.size(1), self.n_head, ()))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut qk = (q * scale)?.matmul(&(k * scale)?.transpose(D::Minus1, D::Minus2)?)?;

        if let Some(mask) = mask {
            qk = qk.broadcast_add(&mask.i((..n_ctx, ..n_ctx))?)?;
        }

        let w = qk.softmax(D::Minus1)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten(2, D::Minus1)?;

        Ok(wv)
    }
}

pub struct ResidualAttentionBlock {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: Option<MultiHeadAttention>,
    encoder_attn_layer_norm: Option<LayerNorm>,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl ResidualAttentionBlock {
    pub fn new(
        n_state: usize,
        n_head: usize,
        ffn_dim: usize,
        cross_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(n_state, n_head, vb.pp("self_attn"))?;
        let self_attn_layer_norm = candle_nn::layer_norm(
            n_state,
            LayerNormConfig::default(),
            vb.pp("self_attn_layer_norm"),
        )?;

        let (encoder_attn, encoder_attn_layer_norm) = if cross_attention {
            let encoder_attn = MultiHeadAttention::new(n_state, n_head, vb.pp("encoder_attn"))?;
            let encoder_attn_layer_norm = candle_nn::layer_norm(
                n_state,
                LayerNormConfig::default(),
                vb.pp("encoder_attn_layer_norm"),
            )?;
            (Some(encoder_attn), Some(encoder_attn_layer_norm))
        } else {
            (None, None)
        };

        let fc1 = candle_nn::linear(n_state, ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(ffn_dim, n_state, vb.pp("fc2"))?;
        let final_layer_norm = candle_nn::layer_norm(
            n_state,
            LayerNormConfig::default(),
            vb.pp("final_layer_norm"),
        )?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
        flush_kv_cache: bool,
    ) -> Result<Tensor> {
        let mut residual = x.clone();

        let x = self.self_attn_layer_norm.forward(&residual)?;
        let x = self.self_attn.forward(&x, None, mask, flush_kv_cache)?;
        residual = (residual + &x)?;

        if let (Some(encoder_attn), Some(encoder_attn_layer_norm)) =
            (&mut self.encoder_attn, &self.encoder_attn_layer_norm)
        {
            let x = encoder_attn_layer_norm.forward(&residual)?;
            let x = encoder_attn.forward(&x, xa, None, flush_kv_cache)?;
            residual = (residual + &x)?;
        }

        let x = self.final_layer_norm.forward(&residual)?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.fc2.forward(&x)?;
        residual = (residual + &x)?;

        Ok(residual)
    }
}
