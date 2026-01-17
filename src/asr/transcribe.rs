use anyhow::Result;
use candle::{DType, Device, safetensors};
use candle_nn::VarBuilder;

use super::{parakeet::Parakeet, reazonspeech::ReazonSpeech, silero_vad::SileroVad};

pub struct Transcriber {
    sample_rate: u32,
    samples: Vec<f32>,
    buf_len: usize,
    device: Device,
    vad: SileroVad,
    model: Model,
    speech_count: usize,
    silence_count: usize,
}

impl Transcriber {
    pub fn new(sample_rate: u32, model_name: &str) -> Result<Self> {
        let device = Device::new_cuda(0)?;

        let vad = {
            let tensors = safetensors::load("silero_vad.safetensors", &device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            SileroVad::new(sample_rate, vb)?
        };
        let buf_len = vad.chunk_len();

        let model = Model::new(sample_rate, device.clone(), model_name)?;

        Ok(Self {
            sample_rate,
            samples: vec![0.0; buf_len],
            buf_len,
            device,
            vad,
            model,
            speech_count: 0,
            silence_count: 0,
        })
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<(Option<String>, Option<String>)> {
        let mut conformed_text = None;
        let mut tentative_text = None;

        self.samples.extend_from_slice(samples);
        let chunk_len = self.vad.chunk_len();
        let n_chunks = (self.samples.len() - self.buf_len) / chunk_len;

        for i in 0..n_chunks {
            let speech_prob_threshold = self.speech_prob_threshold();
            let silence_length_threshold = self.silence_length_threshold();

            let beg = self.buf_len + i * chunk_len;
            let end = beg + chunk_len;
            let chunk = &self.samples[beg..end];
            let prob = self.vad.infer(chunk)?;
            let speaking = prob > speech_prob_threshold;

            if speaking {
                self.silence_count = 0;

                let chunk = if self.speech_count == 0 {
                    &self.samples[beg - self.buf_len..end]
                } else {
                    chunk
                };
                self.model.push(chunk)?;
                self.speech_count += 1;
            } else {
                if self.silence_count < silence_length_threshold {
                    self.model.push(chunk)?;
                    self.speech_count += 1;
                } else if self.speech_count > 0 {
                    conformed_text = self.model.transcribe()?;
                    self.model.clear()?;
                    self.speech_count = 0;
                }

                self.silence_count += 1;
            }
        }

        if conformed_text.is_none() && self.speech_count > 30 {
            tentative_text = self.model.transcribe()?;
        }

        let drain_len = n_chunks * chunk_len;
        let remain_len = self.samples.len() - drain_len;
        self.samples.copy_within(drain_len.., 0);
        self.samples.resize(remain_len, 0.0);

        Ok((conformed_text, tentative_text))
    }

    pub fn reset(&mut self) -> Result<()> {
        self.vad.reset()?;
        self.model.clear()?;
        self.samples.clear();
        self.samples.resize(self.buf_len, 0.0);
        self.speech_count = 0;
        self.silence_count = 0;
        Ok(())
    }

    pub fn set_model(&mut self, model_name: &str) -> Result<()> {
        self.model = Model::new(self.sample_rate, self.device.clone(), model_name)?;
        self.speech_count = 0;
        self.silence_count = 0;
        Ok(())
    }

    fn speech_prob_threshold(&self) -> f32 {
        if self.speech_count < 30 {
            // ~1sec
            0.9
        } else if self.speech_count < 60 {
            // ~2sec
            0.95
        } else if self.speech_count < 150 {
            // ~5sec
            0.98
        } else if self.speech_count < 300 {
            // ~10sec
            0.99
        } else if self.speech_count < 600 {
            // ~20sec
            0.995
        } else if self.speech_count < 900 {
            // ~30sec
            0.999
        } else {
            1.0
        }
    }

    fn silence_length_threshold(&self) -> usize {
        if self.speech_count < 30 {
            // ~1sec
            20
        } else if self.speech_count < 60 {
            // ~2sec
            15
        } else if self.speech_count < 150 {
            // ~5sec
            12
        } else if self.speech_count < 300 {
            // ~10sec
            10
        } else if self.speech_count < 600 {
            // ~20sec
            5
        } else if self.speech_count < 900 {
            // ~30sec
            1
        } else {
            0
        }
    }
}

enum Model {
    None,
    Parakeet(Box<Parakeet>),
    ReazonSpeech(Box<ReazonSpeech>),
}

impl Model {
    fn new(sample_rate: u32, device: Device, model_name: &str) -> Result<Self> {
        match model_name {
            "parakeet" => Self::new_parakeet(sample_rate, device),
            "reazonspeech" => Self::new_reazonspeech(sample_rate, device),
            _ => Ok(Self::None),
        }
    }

    fn new_parakeet(sample_rate: u32, device: Device) -> Result<Self> {
        let parakeet = {
            let tensors = safetensors::load("parakeet-tdt-0.6b-v2.safetensors", &device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            Parakeet::new(sample_rate, vb)?
        };
        Ok(Self::Parakeet(Box::new(parakeet)))
    }

    fn new_reazonspeech(sample_rate: u32, device: Device) -> Result<Self> {
        let reazonspeech = {
            let tensors = safetensors::load("reazonspeech-k2-v2.safetensors", &device)?;
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            ReazonSpeech::new(sample_rate, vb)?
        };
        Ok(Self::ReazonSpeech(Box::new(reazonspeech)))
    }

    fn push(&mut self, samples: &[f32]) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Parakeet(parakeet) => parakeet.push(samples),
            Self::ReazonSpeech(reazonspeech) => reazonspeech.push(samples),
        }
    }

    fn transcribe(&mut self) -> Result<Option<String>> {
        match self {
            Self::None => Ok(None),
            Self::Parakeet(parakeet) => parakeet.transcribe(),
            Self::ReazonSpeech(reazonspeech) => reazonspeech.transcribe(),
        }
    }

    fn clear(&mut self) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Parakeet(parakeet) => parakeet.clear(),
            Self::ReazonSpeech(reazonspeech) => reazonspeech.clear(),
        }
    }
}
