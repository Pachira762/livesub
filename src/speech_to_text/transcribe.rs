use anyhow::Result;
use candle::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, model::Whisper, Config};
use hf_hub::{api::sync::Api, Repo};
use tokenizers::Tokenizer;

use super::mel::MelSpectrogram;

pub struct Transcriber {
    device: Device,
    config: Config,

    model: Whisper,
    suppress_tokens: Tensor,

    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    initial_tokens: Vec<u32>,
    interrupt_tokens: Vec<u32>,

    melspec: MelSpectrogram,
}

impl Transcriber {
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
                Tokenizer::from_file(tokenizer).map_err(anyhow::Error::msg)?,
            )
        };

        let suppress_tokens = {
            let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
                .map(|i| {
                    if config.suppress_tokens.contains(&i) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect();

            Tensor::new(suppress_tokens, &device)?
        };

        let initial_tokens = vec![
            tokenizer.token_to_id(m::SOT_TOKEN).unwrap(),
            tokenizer.token_to_id(m::TRANSCRIBE_TOKEN).unwrap(),
            tokenizer.token_to_id(m::NO_TIMESTAMPS_TOKEN).unwrap(),
        ];

        let mut interrupt_tokens = vec![tokenizer.token_to_id(m::EOT_TOKEN).unwrap()];
        if let Some(token) = tokenizer.token_to_id(m::NO_SPEECH_TOKENS[0]) {
            interrupt_tokens.push(token);
        }
        if let Some(token) = tokenizer.token_to_id(m::NO_SPEECH_TOKENS[1]) {
            interrupt_tokens.push(token);
        }

        let melspec = MelSpectrogram::new(config.num_mel_bins)?;

        Ok(Self {
            device,
            config,
            model,
            suppress_tokens,
            tokenizer,
            tokens: vec![],
            initial_tokens,
            interrupt_tokens,
            melspec,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<Option<(String, bool)>> {
        let (features, is_new_segment) = if let Some((mel, is_new_segment)) =
            self.melspec.decode(audio)
        {
            let mel_len = mel.len();
            let num_mel_bins = self.config.num_mel_bins;
            let mel =
                Tensor::from_slice(mel, (1, num_mel_bins, mel_len / num_mel_bins), &self.device)?;
            let features = self.model.encoder.forward(&mel, is_new_segment)?;
            (features, is_new_segment)
        } else {
            return Ok(None);
        };

        if is_new_segment || self.tokens.is_empty() {
            self.init_tokens();
        } else {
            self.forget_tokens(4);
        }

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

            if self.interrupt_tokens.contains(&next_token) {
                break;
            }

            self.tokens.push(next_token);

            if self.tokens.len() > self.config.max_target_positions {
                break;
            }
        }

        let text = self
            .tokenizer
            .decode(&self.tokens, true)
            .map_err(anyhow::Error::msg)?;

        Ok(Some((text, is_new_segment)))
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.melspec.clear();
    }

    fn init_tokens(&mut self) {
        self.tokens = self.initial_tokens.clone();
    }

    fn forget_tokens(&mut self, n_forget: usize) {
        let n_initial = self.initial_tokens.len();
        let len = self.tokens.len().saturating_sub(n_forget).max(n_initial);
        self.tokens.truncate(len);
    }
}
