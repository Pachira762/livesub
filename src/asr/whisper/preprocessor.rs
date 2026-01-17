use candle::{Device, Result, Tensor};

use crate::asr::common::mel::{MelFilterType, MelProcessor, WindowFunc};

pub struct LogMelSpectrogramPreprocessor {
    device: Device,
    hop_len: usize,
    window_len: usize,
    n_mels: usize,
    mel_processor: MelProcessor,
    samples: Vec<f32>,
}

impl LogMelSpectrogramPreprocessor {
    pub fn new(sample_rate: u32, n_mels: usize, device: &Device) -> Result<Self> {
        let hop_len = 160 * sample_rate as usize / 16000;
        let window_len = 400 * sample_rate as usize / 16000;
        let n_fft = 400 * sample_rate as usize / 16000;
        let mel_processor = MelProcessor::new(
            0.0,
            WindowFunc::Hann,
            window_len,
            n_fft,
            MelFilterType::Librosa,
            n_mels,
            sample_rate,
            0.0,
            8000.0,
        )?;

        Ok(Self {
            device: device.clone(),
            hop_len,
            window_len,
            n_mels,
            mel_processor,
            samples: vec![],
        })
    }

    /// input: x(T)
    /// output: x(B, T/100, F)
    pub fn process(&mut self, audio: &[f32]) -> Result<Option<Tensor>> {
        self.samples.extend_from_slice(audio);
        if self.samples.len() < self.window_len {
            return Ok(None);
        }

        let n_mels = self.n_mels;
        let n_frames = (self.samples.len() - self.window_len) / self.hop_len + 1;
        let mut frames = vec![0.0; n_mels * n_frames];

        for i in 0..n_frames {
            let start = i * self.hop_len;
            let end = start + self.window_len;
            let audio = &self.samples[start..end];
            let mels = self.mel_processor.process(audio);

            for j in 0..n_mels {
                frames[j * n_frames + i] = mels[j];
            }
        }

        let x = Tensor::from_slice(&frames, (1, n_mels, n_frames), &self.device)?;
        let x = (x.maximum(1e-10)?.log()? / 10.0f64.ln())?;
        let x = x.broadcast_maximum(&(x.max_all()? - 8.0)?)?;
        let x = ((x + 4.0)? / 4.0)?;

        // // const SAMPLE_RATE: usize = 16000;
        // const HOP_LEN: usize = 160;
        // const WINDOW_LEN: usize = 400;
        // const CHUNKS: usize = 2864;
        // const N_SAMPLES: usize = HOP_LEN * (CHUNKS - 1) + WINDOW_LEN;

        // if self.samples.len() > N_SAMPLES {
        //     let n_hop = CHUNKS * HOP_LEN;
        //     self.samples = self.samples[n_hop..].to_vec();
        // }
        // self.samples.extend_from_slice(audio);

        // let audio = if self.samples.len() < N_SAMPLES {
        //     // let n_pad = N_SAMPLES - self.samples.len();
        //     let n_pad = WINDOW_LEN;
        //     [self.samples.to_vec(), vec![0.0; n_pad]].concat()
        // } else {
        //     println!("clip input");
        //     self.samples[..N_SAMPLES].to_vec()
        // };

        // let n_frames = (audio.len() - WINDOW_LEN) / HOP_LEN + 1;
        // println!("process {n_frames} frame");
        // let n_mels = self.n_mels;
        // let mut feature = vec![0.0f32; n_frames * n_mels];

        // for i in 0..n_frames {
        //     self.process_frame(&audio, i);

        //     for j in 0..n_mels {
        //         feature[j * n_frames + i] = self.mel[j];
        //     }
        // }

        // let x = Tensor::new(feature, &self.device)?
        //     .reshape((n_mels, n_frames))?
        //     .unsqueeze(0)?;

        // let x = x.broadcast_maximum(&(x.max_all()? - 8.0)?)?;
        // let x = ((x + 4.0)? / 4.0)?;

        Ok(Some(x))
    }

    // fn process_frame(&mut self, audio: &[f32], frame_index: usize) {
    //     const PREEMPH: f32 = 0.97;

    //     let start = frame_index * self.hop_len;
    //     let end = (start + self.window_len).min(audio.len());
    //     let len = end - start;

    //     for i in 0..len {
    //         let x = audio[start + i] - PREEMPH * self.prev;
    //         self.prev = audio[start + i];
    //         self.fft_in[i] = self.window[i] * x;
    //     }
    //     for i in len..self.n_fft {
    //         self.fft_in[i] = 0.0;
    //     }

    //     self.fft
    //         .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut self.fft_scratch)
    //         .unwrap();

    //     for (power, c) in self.power.iter_mut().zip(&self.fft_out) {
    //         *power = c.norm_sqr();
    //     }

    //     for (i, mel) in self.mel.iter_mut().enumerate() {
    //         let (offset, coeffs) = &self.melbank[i];
    //         *mel = coeffs
    //             .iter()
    //             .zip(&self.power[*offset..])
    //             .map(|(&coeff, &power)| coeff * power)
    //             .sum::<f32>()
    //             .max(1e-10)
    //             .log10();
    //     }
    // }
}
