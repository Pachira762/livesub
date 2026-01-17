mod attention;
mod config;
mod decoder;
mod encoder;
mod preprocessor;

use anyhow::{Error, Result};
use candle::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{Repo, api::sync::Api};
use tokenizers::Tokenizer;

use crate::asr::whisper::{
    config::Config, decoder::TextDecoder, encoder::AudioEncoder,
    preprocessor::LogMelSpectrogramPreprocessor,
};

pub struct Whisper {
    device: Device,
    config: Config,
    preprocessor: LogMelSpectrogramPreprocessor,
    encoder: AudioEncoder,
    decoder: TextDecoder,
    tokenizer: Tokenizer,
}

impl Whisper {
    pub fn new(repo_id: &str, sample_rate: u32) -> Result<Self> {
        let device = Device::new_cuda(0)?;

        let (config, tokenizer, model) = {
            let api = Api::new()?;
            let repo = api.repo(Repo::model(repo_id.to_string()));
            (
                repo.get("config.json")?,
                repo.get("tokenizer.json")?,
                repo.get("model.safetensors")?,
            )
        };

        let model = candle::safetensors::load(model, &device)?;
        let vb = VarBuilder::from_tensors(model, DType::F32, &device);
        let config = Config::from_file(&config)?;
        let preprocessor =
            LogMelSpectrogramPreprocessor::new(sample_rate, config.num_mel_bins, &device)?;
        let encoder = AudioEncoder::new(&config, vb.pp("model.encoder"))?;
        let decoder = TextDecoder::new(&config, vb.pp("model.decoder"))?;
        let tokenizer = Tokenizer::from_file(&tokenizer).map_err(Error::msg)?;

        Ok(Self {
            device,
            config,
            preprocessor,
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<Option<String>> {
        let bos = self.tokenizer.token_to_id("<|startoftranscript|>").unwrap();
        let lang = self.tokenizer.token_to_id("<|en|>").unwrap();
        let task = self.tokenizer.token_to_id("<|transcribe|>").unwrap();
        let no_timestamp = self.tokenizer.token_to_id("<|notimestamps|>").unwrap();
        let eot = self.tokenizer.token_to_id("<|endoftext|>").unwrap();
        let spress_tokens = self
            .config
            .suppress_tokens
            .as_ref()
            .cloned()
            .unwrap_or_default();
        let spress: Vec<_> = (0..self.config.vocab_size)
            .map(|i| {
                if spress_tokens.contains(&(i as u32)) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let spress = Tensor::new(spress, &self.device)?;

        let feats = self.preprocessor.process(audio)?.unwrap();
        let encoder_out = self.encoder.forward(&feats)?;

        let mut tokens: Vec<u32> = vec![bos, lang, task, no_timestamp];
        let (_batch, seq_len, _hidden) = encoder_out.dims3()?;
        for i in 0..seq_len {
            let token_t = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            let y = self.decoder.forward(&token_t, &encoder_out, i == 0)?;
            let token = y
                .i((0, 0, ..))?
                .add(&spress)?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;
            if token == eot {
                break;
            }
            tokens.push(token);
        }

        let text = self.tokenizer.decode(&tokens, true).unwrap();

        Ok(Some(text))
    }
}
