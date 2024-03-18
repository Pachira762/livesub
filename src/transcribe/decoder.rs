use anyhow::{Error as E, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, model::Whisper, Config};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

pub struct Decoder {
    device: Device,
    model: Whisper,
    config: Config,
    tokenizer: Tokenizer,
    initial_tokens: Vec<u32>,
    eot_token: u32,
    suppress_tokens: Tensor,
    tokens: Vec<u32>,
}

impl Decoder {
    pub fn new(repo_id: &str) -> Result<Self> {
        let device = Device::new_cuda(0)?;

        let (model, config, tokenizer) = {
            let api = Api::new()?;
            let repo = api.repo(Repo::new(repo_id.to_owned(), hf_hub::RepoType::Model));

            let (model, config, tokenizer) = (
                repo.get("model.safetensors")?,
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
            );

            let config: Config = serde_json::from_str(&std::fs::read_to_string(config)?)?;
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], m::DTYPE, &device)? };

            (
                m::model::Whisper::load(&vb, config.clone())?,
                config,
                Tokenizer::from_file(tokenizer).map_err(E::msg)?,
            )
        };

        let (sot_token, task_token, no_timestamps_token, eot_token) = {
            (
                token_id(&tokenizer, m::SOT_TOKEN)?,
                token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?,
                token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?,
                token_id(&tokenizer, m::EOT_TOKEN)?,
            )
        };

        let initial_tokens = vec![sot_token, task_token, no_timestamps_token];

        let suppress_tokens = {
            let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
                .map(|i| {
                    if config.suppress_tokens.contains(&i) || i == no_timestamps_token {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
                .collect();

            Tensor::new(suppress_tokens, &device)?
        };

        Ok(Self {
            device,
            model,
            config,
            tokenizer,
            initial_tokens,
            eot_token,
            suppress_tokens,
            tokens: vec![],
        })
    }

    pub fn process(&mut self, mel: &[f32], new_segment: bool) -> Result<String> {
        if new_segment || self.tokens.is_empty() {
            self.tokens = self.initial_tokens.clone();
        } else {
            self.forget_tokens(2);
        }

        let mel = {
            let mel_len = mel.len();
            let num_mel_bins = self.config.num_mel_bins;
            Tensor::from_slice(mel, (1, num_mel_bins, mel_len / num_mel_bins), &self.device)?
        };

        self.decode(&mel, new_segment)?;

        let text = self.tokenizer.decode(&self.tokens, true).map_err(E::msg)?;
        Ok(text)
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
    }

    fn decode(&mut self, mel: &Tensor, flush: bool) -> Result<()> {
        let features = self.model.encoder.forward(mel, flush)?;

        for i in 0.. {
            let tokens_t = Tensor::new(self.tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let ys = self.model.decoder.forward(&tokens_t, &features, i == 0)?;

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?
                .broadcast_add(&self.suppress_tokens)?;

            let next_token = logits
                .to_vec1::<f32>()?
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as u32)
                .unwrap();

            if next_token == self.eot_token {
                break;
            }

            self.tokens.push(next_token);

            if self.tokens.len() > self.config.max_target_positions {
                break;
            }
        }

        Ok(())
    }

    fn forget_tokens(&mut self, n_forget: usize) {
        let n_remain = self.tokens.len().saturating_sub(n_forget);
        let n_init: usize = self.initial_tokens.len();
        self.tokens.truncate(n_remain.max(n_init));
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}
