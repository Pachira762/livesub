mod attention;
mod conformer;
mod decoder;
mod encoder;
mod greedy_search;
mod joiner;
mod preprocessor;
mod subsampling;
mod tokenizer;

use anyhow::Result;
use candle_nn::VarBuilder;

use crate::asr::parakeet::{
    encoder::ConformerEncoder, greedy_search::GreedyTdtDecoder,
    preprocessor::AudioToMelSpectrogramPreprocessor, tokenizer::Tokenizer,
};

pub struct Parakeet {
    #[allow(unused)]
    sample_rate: u32,
    preprocessor: AudioToMelSpectrogramPreprocessor,
    encoder: ConformerEncoder,
    decoder: GreedyTdtDecoder,
    tokenizer: Tokenizer,
}

impl Parakeet {
    pub fn new(sample_rate: u32, vb: VarBuilder) -> Result<Self> {
        let mut preprocessor =
            AudioToMelSpectrogramPreprocessor::new(sample_rate, vb.device().clone())?;
        preprocessor.push(&vec![0.0; sample_rate as usize / 100])?;

        let encoder = ConformerEncoder::new(vb.pp("encoder"))?;
        let decoder = GreedyTdtDecoder::new(vb)?;
        let tokenizer = Tokenizer::new()?;

        Ok(Self {
            sample_rate,
            preprocessor,
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn push(&mut self, audio: &[f32]) -> Result<()> {
        self.preprocessor.push(audio)?;
        Ok(())
    }

    pub fn transcribe(&mut self) -> Result<Option<String>> {
        let x = match self.preprocessor.process()? {
            Some(x) => x,
            None => return Ok(None),
        };

        let x = match self.encoder.forward(&x)? {
            Some(x) => x,
            None => return Ok(None),
        };

        let tokens = self.decoder.infer(&x)?;
        let text = self.tokenizer.ids_to_text(&tokens);

        Ok(Some(text))
    }

    pub fn clear(&mut self) -> Result<()> {
        self.preprocessor.clear()?;
        self.encoder.clear()?;
        self.decoder.clear()?;

        Ok(())
    }
}
