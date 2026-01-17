mod decoder;
mod encoder;
mod greedy_search;
mod joiner;
mod preprocessor;
mod scaling;
mod subsampling;
mod tokenizer;
mod zipformer;

use anyhow::Result;
use candle_nn::VarBuilder;

use crate::asr::reazonspeech::{
    encoder::Encoder, greedy_search::GreedySearchInfer, preprocessor::FeatureExtractor,
    tokenizer::Tokenizer,
};

pub struct ReazonSpeech {
    sample_rate: u32,
    preprocessor: FeatureExtractor,
    encoder: Encoder,
    decoder: GreedySearchInfer,
    tokenizer: Tokenizer,
}

impl ReazonSpeech {
    pub fn new(sample_rate: u32, vb: VarBuilder) -> Result<Self> {
        let mut preprocessor = FeatureExtractor::new(sample_rate, vb.device().clone())?;
        preprocessor.push(&vec![0.0; sample_rate as usize / 2])?;

        let encoder = Encoder::new(512, vb.pp("encoder"))?;
        let decoder = GreedySearchInfer::new(vb)?;
        let tokenizer = Tokenizer::new()?;

        Ok(Self {
            sample_rate,
            preprocessor,
            encoder,
            decoder,
            tokenizer,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.preprocessor.push(samples)?;
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
        self.preprocessor
            .push(&vec![0.0; self.sample_rate as usize / 2])?;

        self.encoder.clear()?;
        self.decoder.clear()?;
        Ok(())
    }
}

/*
あ
あら
ある朝
あるあさ
ある朝めざめ
ある朝めざめ
ある朝めざめる
ある朝めざめると
ある朝めざめると
ある朝めざめると
ある朝めざめると
ある朝めざめると
ある朝めざめるとう
ある朝めざめるとん
えっ
えっ
えっ
えっ
えっ
えっ
えっ
あっ
あっ
あっ
あっ
あっ
えっ
何
なに
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何
人間
人間像
人間と竜
人間と竜の
人間と竜の母
人間と竜のハーフ
人間と竜のハーフコート
人間と竜のハーフ孝行
人間と竜のハーフ高校生
人間と竜のハーフ高校生の
人間と竜のハーフ高校生の姿
人間と竜のハーフ高校生の救い
人間と竜のハーフ高校生のスクール
人間と竜のハーフ高校生のスクールライブ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフルリ
人間と竜のハーフ高校生のスクールライフルリドラゴン
人間と竜のハーフ高校生のスクールライフルリドラゴン
人間と竜のハーフ高校生のスクールライフルリドラゴン

あ
ある
ある朝
あるあさも
ある朝めざ
ある朝めざめ
ある朝めざめる
ある朝めざめると
ある朝めざめると
ある朝めざめると
ある朝めざめると
ある朝めざめると
ある朝めざめるとう
ある朝めざめるとん
え
え
え
え
え
え
え
そう
あっ
あっ
あっ
あっ
え
ねえ
なに
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何これ角が生えていた
何
人間
人間
人間と竜
人間と竜の
人間と竜の花
人間と竜のハーフ
人間と竜のハーフコート
人間と竜のハーフ公開
人間と竜のハーフ高校生
人間と竜のハーフ高校生
人間と竜のハーフ高校生の姿
人間と竜のハーフ高校生のスクープ
人間と竜のハーフ高校生のスクール
人間と竜のハーフ高校生のスクールライブ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフ
人間と竜のハーフ高校生のスクールライフルリ
人間と竜のハーフ高校生のスクールライフルリドラゴン
人間と竜のハーフ高校生のスクールライフルリドラゴン
人間と竜のハーフ高校生のスクールライフルリドラゴン
*/
