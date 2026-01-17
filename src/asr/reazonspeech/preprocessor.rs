use core::f32;

use candle::{Device, Result, Tensor};

use crate::asr::common::{
    mel::{MelFilterType, MelProcessor},
    window,
};

pub struct FeatureExtractor {
    device: Device,
    mel_processor: MelProcessor,
}

impl FeatureExtractor {
    pub fn new(sample_rate: u32, device: Device) -> Result<Self> {
        let hop_len = 160 * sample_rate as usize / 16000;
        let window_len = 400 * sample_rate as usize / 16000;
        let window = window::povey(window_len);
        let n_fft = if sample_rate == 16000 {
            window_len.next_power_of_two()
        } else {
            window_len
        };
        let n_mels = 80;
        let mel_processor = MelProcessor::new(
            0.97,
            window,
            hop_len,
            n_fft,
            MelFilterType::Kaldi,
            n_mels,
            sample_rate,
            20.0,
            8000.0 - 400.0,
        )?;

        Ok(Self {
            device,
            mel_processor,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<()> {
        self.mel_processor.push(samples);
        Ok(())
    }

    pub fn process(&mut self) -> Result<Option<Tensor>> {
        let mels = self.mel_processor.mels();
        if mels.is_empty() {
            return Ok(None);
        }

        let n_frames = self.mel_processor.num_frames();
        let n_mels = self.mel_processor.num_mel_bins();
        let x = Tensor::from_slice(mels, (1, n_frames, n_mels), &self.device)?;

        Ok(Some(x))
    }

    pub fn clear(&mut self) -> Result<()> {
        self.mel_processor.clear();

        Ok(())
    }
}
