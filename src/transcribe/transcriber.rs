use anyhow::Result;

use candle_transformers::models::whisper::SAMPLE_RATE;

use super::audio::AudioBuffer;
use super::decoder::Decoder;
use super::mel::{MelDecodeResult, MelSpectrogram};

pub struct TranscribeResult {
    pub text: String,
    pub new_segment: bool,
}

pub struct Transcriber {
    audio: AudioBuffer,
    mel: MelSpectrogram,
    decoder: Decoder,
}

impl Transcriber {
    pub fn new(repo_id: &str) -> Result<Self> {
        let audio = AudioBuffer::new(SAMPLE_RATE as u32)?;

        let decoder = Decoder::new(repo_id)?;

        let num_mel_bins = decoder.config().num_mel_bins;
        let mel = MelSpectrogram::new(num_mel_bins)?;

        Ok(Self {
            audio,
            mel,
            decoder,
        })
    }

    pub fn transcribe(&mut self) -> Result<Option<TranscribeResult>> {
        let pcm = self.audio.capture()?;

        let MelDecodeResult {
            n_frames,
            new_segment,
        } = self.mel.process(pcm);

        if n_frames > 0 {
            let text = self.decoder.process(self.mel.mel(), new_segment)?;
            Ok(Some(TranscribeResult { text, new_segment }))
        } else {
            Ok(None)
        }
    }

    pub fn set_model(&mut self, repo_id: &str) -> Result<()> {
        let decoder = Decoder::new(repo_id)?;

        let num_mel_bins = decoder.config().num_mel_bins;
        let mel = MelSpectrogram::new(num_mel_bins)?;

        self.decoder = decoder;
        self.mel = mel;

        Ok(())
    }

    pub fn clear(&mut self) {
        self.mel.clear();
        self.decoder.clear();
    }
}
