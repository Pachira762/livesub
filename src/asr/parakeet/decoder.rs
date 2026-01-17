use crate::asr::common::tensor_ext::TensorExt;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LSTM, LSTMConfig, RNN, VarBuilder, rnn::LSTMState};

pub struct RnntDecoder {
    device: Device,
    embed: Embedding,
    dec_rnn: StackedLstm,
}

impl RnntDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let input_size = 1024 + 1;
        let embedding_size = 640;
        let output_size = 640;
        let embed = candle_nn::embedding(input_size, embedding_size, vb.pp("prediction.embed"))?;
        let dec_rnn =
            StackedLstm::new(embedding_size, output_size, 2, vb.pp("prediction.dec_rnn"))?;

        Ok(Self {
            device,
            embed,
            dec_rnn,
        })
    }

    pub fn predict(
        &self,
        token: u32,
        state: Option<Vec<LSTMState>>,
    ) -> Result<(Tensor, Vec<LSTMState>)> {
        let x = if token == 1024 {
            let hidden = self.embed.hidden_size();
            Tensor::zeros((1, 1, hidden), DType::F32, &self.device)?
        } else {
            let x = Tensor::new(&[[token]], &self.device)?;
            self.embed.forward(&x)?
        };
        let x = x.transpose(0, 1)?;

        let (y, state) = self.dec_rnn.forward(&x, state)?;
        let y = y.transpose(0, 1)?;

        Ok((y, state))
    }
}

struct StackedLstm {
    layers: Vec<LSTM>,
}

impl StackedLstm {
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layers: Vec<_> = (0..num_layers)
            .map(|i| {
                candle_nn::lstm(
                    input_size,
                    hidden_size,
                    LSTMConfig {
                        layer_idx: i,
                        ..Default::default()
                    },
                    vb.pp("lstm"),
                )
            })
            .collect::<Result<_>>()?;

        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor,
        state: Option<Vec<LSTMState>>,
    ) -> Result<(Tensor, Vec<LSTMState>)> {
        let mut state = match state {
            Some(state) => state,
            None => self.initial_state()?,
        };

        let ys: Vec<_> = (0..xs.size(0))
            .map(|i| {
                let mut x = xs.get(i)?;
                for (j, layer) in self.layers.iter().enumerate() {
                    state[j] = layer.step(&x, &state[j])?;
                    x = state[j].h().clone();
                }
                Ok(x)
            })
            .collect::<Result<_>>()?;

        let ys = Tensor::stack(&ys, 0)?;
        Ok((ys, state))
    }

    fn initial_state(&self) -> Result<Vec<LSTMState>> {
        self.layers
            .iter()
            .map(|lstm| lstm.zero_state(1))
            .collect::<Result<_>>()
    }
}
