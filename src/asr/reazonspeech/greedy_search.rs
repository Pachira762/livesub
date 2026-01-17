use candle::{D, Device, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

use crate::asr::{
    common::tensor_ext::TensorExt,
    reazonspeech::{decoder::Decoder, joiner::Joiner},
};

pub struct GreedySearchInfer {
    device: Device,
    decoder: Decoder,
    joiner: Joiner,
    cache: Cache,
}

impl GreedySearchInfer {
    const BLANK: u32 = 0;
    const UNK: u32 = 5222;

    pub fn new(vb: VarBuilder) -> Result<Self> {
        let decoder_dim = 512;
        let joiner_dim = 512;
        let vocab_size = 5224;
        let context_size = 2;
        let decoder = Decoder::new(
            vocab_size,
            decoder_dim,
            context_size,
            joiner_dim,
            vb.pp("decoder"),
        )?;
        let joiner = Joiner::new(joiner_dim, vocab_size, vb.pp("joiner"))?;

        Ok(Self {
            device: vb.device().clone(),
            decoder,
            joiner,
            cache: Cache::new(),
        })
    }

    pub fn infer(&mut self, encoder_out: &Tensor) -> Result<Vec<u32>> {
        let (mut tokens, start_time) = self.cache.resume();
        let mut decoder_out = self.decode(&tokens)?;

        let seq_len = encoder_out.size(1);
        for time in start_time..seq_len {
            let encoder_out = encoder_out.narrow(1, time, 1)?;

            for _ in 0..16 {
                let logits = self
                    .joiner
                    .forward(&encoder_out, &decoder_out)?
                    .i((0, 0))?
                    .softmax(D::Minus1)?;
                let token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

                if token != Self::BLANK && token < Self::UNK {
                    tokens.push(token);

                    let score = logits.i(token as usize)?.to_scalar::<f32>()?;
                    self.cache.update(token, score, time);

                    decoder_out = self.decode(&tokens)?;
                    continue;
                }

                break;
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<Tensor> {
        let context = if tokens.is_empty() {
            vec![Self::UNK, Self::BLANK]
        } else if tokens.len() == 1 {
            vec![Self::BLANK, tokens[0]]
        } else {
            tokens.last_chunk::<2>().unwrap().to_vec()
        };
        let input = Tensor::from_vec(context, (1, 2), &self.device)?;
        let output = self.decoder.forward(&input)?;
        Ok(output)
    }

    pub fn clear(&mut self) -> Result<()> {
        self.cache.clear();
        Ok(())
    }
}

struct Cache {
    tokens: Vec<u32>,
    scores: Vec<f32>,
    times: Vec<usize>,
}

impl Cache {
    fn new() -> Self {
        Self {
            tokens: vec![],
            scores: vec![],
            times: vec![],
        }
    }

    fn resume(&mut self) -> (Vec<u32>, usize) {
        while self.scores.last().is_some_and(|&score| score < 0.9) {
            _ = self.tokens.pop();
            _ = self.scores.pop();
            _ = self.times.pop();
        }

        for _ in 0..8 {
            _ = self.tokens.pop();
            _ = self.scores.pop();
            _ = self.times.pop();
        }

        (self.tokens.clone(), self.times.last().cloned().unwrap_or(0))
    }

    fn update(&mut self, token: u32, score: f32, time: usize) {
        self.tokens.push(token);
        self.scores.push(score);
        self.times.push(time);
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.scores.clear();
        self.times.clear();
    }
}

// struct Cache {
//     tokens: Vec<u32>,
//     counts: Vec<u32>,
//     token_index: usize,

//     cached_token_index: usize,
//     next_encode_time: usize,
//     should_cache: bool,
// }

// impl Cache {
//     fn new() -> Self {
//         Self {
//             tokens: vec![],
//             counts: vec![],
//             token_index: 0,
//             cached_token_index: 0,
//             next_encode_time: 0,
//             should_cache: true,
//         }
//     }

//     fn resume(&mut self) -> (Vec<u32>, usize) {
//         self.should_cache = true;

//         if self.tokens.len() < 3 {
//             self.tokens.clear();
//             self.counts.clear();
//             self.token_index = 0;
//             self.cached_token_index = 0;
//             self.next_encode_time = 0;
//             (vec![], 0)
//         } else {
//             self.token_index = self.cached_token_index;

//             let tokens = self.tokens[0..self.token_index].to_vec();
//             let time = self.next_encode_time;
//             (tokens, time)
//         }
//     }

//     fn update(&mut self, token: u32, score: f32, time: usize) {
//         if self.token_index < self.tokens.len() {
//             if self.tokens[self.token_index] == token {
//                 self.counts[self.token_index] += 1;

//                 if self.counts[self.token_index] < 5 || score < 0.9 {
//                     self.should_cache = false;
//                 }
//             } else {
//                 self.tokens[self.token_index] = token;
//                 self.counts[self.token_index] = 0;
//                 self.should_cache = false;
//             }
//         } else {
//             self.tokens.push(token);
//             self.counts.push(0);
//             self.should_cache = false;
//         }

//         self.token_index += 1;

//         if self.should_cache {
//             self.cached_token_index = self.token_index;
//             self.next_encode_time = time;
//         }
//     }

//     fn clear(&mut self) {
//         self.tokens.clear();
//         self.counts.clear();
//         self.token_index = 0;
//         self.cached_token_index = 0;
//         self.next_encode_time = 0;
//         self.should_cache = false;
//     }
// }
