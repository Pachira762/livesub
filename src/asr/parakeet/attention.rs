use candle::{D, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::asr::common::tensor_ext::TensorExt;

const INF_VAL: f64 = 10000.0;

pub struct RelPositionEncoding {
    d_model: usize,
    pe: Tensor,
}

impl Module for RelPositionEncoding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let input_len = xs.size(1);
        let center_pos = self.pe.size(1) / 2 + 1;
        let start_pos = center_pos - input_len;
        let end_pos = center_pos + input_len - 1;
        self.pe.i((.., start_pos..end_pos))
    }
}

impl RelPositionEncoding {
    pub fn new(length: usize, device: &Device) -> Result<Self> {
        let mut this = Self {
            d_model: 1024,
            pe: Tensor::new(0.0f32, device)?,
        };
        this.extend_pe(length, device)?;
        Ok(this)
    }

    pub fn extend_pe(&mut self, length: usize, device: &Device) -> Result<()> {
        let length = length as i64;
        let positions = Tensor::arange_step(length - 1, -length, -1, device)?
            .float()?
            .unsqueeze(1)?;
        self.create_pe(&positions, device)
    }

    fn create_pe(&mut self, positions: &Tensor, device: &Device) -> Result<()> {
        let d_model = self.d_model as i64;
        let div_term = Tensor::exp(
            &Tensor::arange_step(0, d_model, 2, device)?
                .float()?
                .scalar_mul(-INF_VAL.ln() / d_model as f64)?,
        )?;

        let p_x_d = positions.broadcast_mul(&div_term)?;
        let (time, feat) = p_x_d.dims2()?;
        let pe_sin = p_x_d.sin()?;
        let pe_cos = p_x_d.cos()?;

        self.pe = Tensor::stack(&[pe_sin, pe_cos], 2)?
            .reshape((time, 2 * feat))?
            .unsqueeze(0)?;

        Ok(())
    }
}

pub struct RelPositionMultiHeadAttention {
    h: usize,
    d_k: usize,
    s_d_k: f64,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
}

impl RelPositionMultiHeadAttention {
    pub fn new(n_head: usize, n_feat: usize, vb: VarBuilder) -> Result<Self> {
        let d_k = n_feat / n_head;
        let s_d_k = (d_k as f64).sqrt();
        let h = n_head;
        let pos_bias_u = vb.get((n_head, d_k), "pos_bias_u")?;
        let pos_bias_v = vb.get((n_head, d_k), "pos_bias_v")?;

        let linear_q = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_out"))?;
        let linear_pos = candle_nn::linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?;

        Ok(Self {
            h,
            d_k,
            s_d_k,
            pos_bias_u,
            pos_bias_v,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let (q, k, v) = self.forward_qkv(query, key, value)?;
        let q = q.transpose(1, 2)?;

        let n_batch_pos = pos_emb.size(0);
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((n_batch_pos, (), self.h, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        let pos_bias_u = self.pos_bias_u.expand(q.shape())?;
        let pos_bias_v = self.pos_bias_v.expand(q.shape())?;
        let q_with_bias_u = (&q + &pos_bias_u)?.transpose(1, 2)?.contiguous()?;
        let q_with_bias_v = (&q + &pos_bias_v)?.transpose(1, 2)?.contiguous()?;

        let matrix_bd = q_with_bias_v.matmul(&p.transpose(D::Minus2, D::Minus1)?)?;
        let matrix_bd = self.rel_shift(&matrix_bd)?;

        let matrix_ac = q_with_bias_u.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let matrix_bd = matrix_bd.i((.., .., .., ..matrix_ac.size(3)))?;
        let scores = (&matrix_ac + &matrix_bd)?.scalar_div(self.s_d_k)?;

        let x = self.forward_attention(&v, &scores)?;

        Ok(x)
    }

    fn forward_attention(&self, value: &Tensor, scores: &Tensor) -> Result<Tensor> {
        let n_batch = value.size(0);

        let attn: Tensor = candle_nn::ops::softmax(scores, D::Minus1)?;
        let x = attn.matmul(value)?;
        let x = x
            .transpose(1, 2)?
            .reshape((n_batch, (), self.h * self.d_k))?;

        let x = self.linear_out.forward(&x)?;

        Ok(x)
    }

    fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let b = query.size(0);

        let q = self
            .linear_q
            .forward(query)?
            .reshape((b, (), self.h, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(key)?
            .reshape((b, (), self.h, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(value)?
            .reshape((b, (), self.h, self.d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        Ok((q, k, v))
    }

    fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, qlen, pos_len) = x.dims4()?;
        x.pad_with_zeros(D::Minus1, 1, 0)?
            .reshape((b, h, (), qlen))?
            .i((.., .., 1..))?
            .reshape((b, h, qlen, pos_len))
    }
}
