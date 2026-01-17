use core::f32;

use candle::{Device, Result, Tensor};

use crate::asr::common::{
    mel::{MelFilterType, MelProcessor},
    tensor_ext::TensorExt,
    window,
};

pub struct AudioToMelSpectrogramPreprocessor {
    device: Device,
    mel_processor: MelProcessor,
}

impl AudioToMelSpectrogramPreprocessor {
    pub fn new(sample_rate: u32, device: Device) -> Result<Self> {
        let hop_len = 160 * sample_rate as usize / 16000;
        let window_len = 400 * sample_rate as usize / 16000;
        let window = window::hanning(window_len);
        let n_fft = if sample_rate == 16000 {
            window_len.next_power_of_two()
        } else {
            window_len
        };
        let n_mels = 128;
        let mel_processor = MelProcessor::new(
            0.97,
            window,
            hop_len,
            n_fft,
            MelFilterType::Librosa,
            n_mels,
            sample_rate,
            0.0,
            8000.0,
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

    /// x(T) => x(B, T/100, F)
    pub fn process(&mut self) -> Result<Option<Tensor>> {
        let mels = self.mel_processor.mels();
        if mels.is_empty() {
            return Ok(None);
        }

        let n_frames = self.mel_processor.num_frames();
        let n_mels = self.mel_processor.num_mel_bins();
        let x = Tensor::from_slice(mels, (1, n_frames, n_mels), &self.device)?;
        let x = self.normalize(&x)?;

        Ok(Some(x))
    }

    // (batch, seq_len, mel_bins)
    fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.size(1) as f64;
        let mean = (x.sum_keepdim(1)? / seq_len)?;
        let std = x
            .broadcast_sub(&mean)?
            .powf(2.0)?
            .sum_keepdim(1)?
            .scalar_div(seq_len - 1.0)?
            .sqrt()?
            .scalar_add(f64::EPSILON)?;
        let x = x.broadcast_sub(&mean)?.broadcast_div(&std)?;

        Ok(x)
    }

    pub fn clear(&mut self) -> Result<()> {
        self.mel_processor.clear();

        Ok(())
    }
}
