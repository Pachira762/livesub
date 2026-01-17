use anyhow::Result;
use candle::{D, Module, Tensor};
use candle_nn::{LSTM, RNN, VarBuilder, rnn::LSTMState};

use crate::asr::common::conv::{Conv1d, Conv1dConfig};

pub struct Decoder {
    rnn: LSTM,
    state: LSTMState,
    final_conv: Conv1d,
}

impl Decoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let in_dim = 128;
        let hidden_dim = 128;
        let config = candle_nn::LSTMConfig {
            ..Default::default()
        };
        let rnn = candle_nn::lstm(in_dim, hidden_dim, config, vb.pp("rnn"))?;
        let state = rnn.zero_state(1)?;

        let in_channels = 128;
        let out_channels = 1;
        let kernel_size = 1;
        let cfg = Conv1dConfig {
            dilation: 1,
            ..Default::default()
        };
        let final_conv = Conv1d::new(
            in_channels,
            out_channels,
            kernel_size,
            cfg,
            vb.pp("final_conv"),
        )?;

        Ok(Self {
            rnn,
            state,
            final_conv,
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.squeeze(D::Minus1)?;
        self.state = self.rnn.step(&xs, &self.state)?;

        let xs = self.state.h().unsqueeze(D::Minus1)?;
        let xs = xs.relu()?;
        let xs = self.final_conv.forward(&xs)?;
        let xs = candle_nn::ops::sigmoid(&xs)?;

        Ok(xs)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.state = self.rnn.zero_state(1)?;
        Ok(())
    }
}
