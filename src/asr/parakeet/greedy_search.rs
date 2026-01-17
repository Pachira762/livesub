use crate::asr::{
    common::tensor_ext::TensorExt,
    parakeet::{decoder::RnntDecoder, joiner::RnntJoiner},
};
use candle::{IndexOp, Result, Tensor};
use candle_nn::{VarBuilder, rnn::LSTMState};

pub struct GreedyTdtDecoder {
    decoder: RnntDecoder,
    joint: RnntJoiner,
    cache: Cache2,
}

impl GreedyTdtDecoder {
    const UNKNOWN: u32 = 0;
    const BLANK: u32 = 1024;
    const DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let decoder = RnntDecoder::new(vb.pp("decoder"))?;
        let joint = RnntJoiner::new(vb.pp("joint"))?;
        Ok(Self {
            decoder,
            joint,
            cache: Cache2::new(),
        })
    }

    pub fn infer(&mut self, encoder_out: &Tensor) -> Result<Vec<u32>> {
        let x = encoder_out.get(0)?.unsqueeze(1)?;

        let (mut tokens, mut state, mut time) = self.cache.resume();
        let seq_len = x.size(0);
        while time < seq_len {
            let feats = x.narrow(0, time, 1)?;
            let mut skip = 0;

            for _ in 0..16 {
                let (token, score) = {
                    let (g, new_state) = self
                        .decoder
                        .predict(*tokens.last().unwrap_or(&Self::BLANK), state)?;
                    state = Some(new_state);

                    let logits = self.joint.joint(&feats, &g)?.i((0, 0, 0))?;
                    let n_logits: usize = logits.size(0);
                    let n_durations = Self::DURATIONS.len();
                    let n_vocab = n_logits - n_durations;

                    let logp = logits.i(..n_vocab)?.softmax(0)?;
                    let token = logp.argmax(0)?.to_scalar::<u32>()?;
                    let score = if token != Self::BLANK {
                        logp.get(token as usize)?.to_scalar::<f32>()?
                    } else {
                        1.0
                    };

                    let duration_logp = logits.i(n_vocab..)?;
                    let duration = duration_logp.argmax(0)?.to_scalar::<u32>()?;
                    skip = Self::DURATIONS[duration as usize];

                    if (token == Self::BLANK || token == Self::UNKNOWN) && skip == 0 {
                        skip = 1;
                    }

                    (token, score)
                };
                time += skip;

                if token != Self::BLANK {
                    tokens.push(token);
                    self.cache
                        .update(token, score, time, state.as_ref().unwrap());
                }

                if skip > 0 {
                    break;
                }
            }

            if skip == 0 {
                time += 1;
            }
        }

        Ok(tokens)
    }

    pub fn clear(&mut self) -> Result<()> {
        self.cache.clear();
        Ok(())
    }
}

struct Cache2 {
    tokens: Vec<u32>,
    scores: Vec<f32>,
    states: Vec<Vec<LSTMState>>,
    times: Vec<usize>,
}

impl Cache2 {
    pub fn new() -> Self {
        Self {
            tokens: vec![],
            scores: vec![],
            states: vec![],
            times: vec![],
        }
    }

    pub fn resume(&mut self) -> (Vec<u32>, Option<Vec<LSTMState>>, usize) {
        const COMMA_ID: u32 = 841;

        while self.scores.last().is_some_and(|&score| score < 0.99)
            || self.tokens.last() == Some(&COMMA_ID)
        {
            _ = self.tokens.pop();
            _ = self.scores.pop();
            _ = self.states.pop();
            _ = self.times.pop();
        }

        for _ in 0..4 {
            _ = self.tokens.pop();
            _ = self.scores.pop();
            _ = self.states.pop();
            _ = self.times.pop();
        }

        if self.tokens.is_empty() {
            (vec![], None, 0)
        } else {
            (
                self.tokens.clone(),
                self.states.last().cloned(),
                *self.times.last().unwrap(),
            )
        }
    }

    pub fn update(&mut self, token: u32, score: f32, time: usize, state: &[LSTMState]) {
        self.tokens.push(token);
        self.scores.push(score);
        self.states.push(state.to_vec());
        self.times.push(time);
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.scores.clear();
        self.states.clear();
        self.times.clear();
    }
}
