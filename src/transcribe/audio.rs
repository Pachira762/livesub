use anyhow::Result;
use rubato::{Resampler, SincFixedOut, SincInterpolationParameters};
use windows::Win32::{
    Media::{Audio::*, Multimedia::WAVE_FORMAT_IEEE_FLOAT},
    System::Com::*,
};

pub struct AudioBuffer {
    capture: AudioCapture,
    resampler: SincFixedOut<f32>,
    buffer: Vec<f32>,
    resampled: Vec<f32>,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let capture = AudioCapture::new()?;
        let resampler = {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                oversampling_factor: 256,
                interpolation: rubato::SincInterpolationType::Linear,
                window: rubato::WindowFunction::BlackmanHarris2,
            };

            let sample_rate_in = capture.sample_rate();
            let sample_rate_out = sample_rate;
            SincFixedOut::<f32>::new(
                sample_rate_out as f64 / sample_rate_in as f64,
                8.0,
                params,
                1024,
                1,
            )
            .map_err(anyhow::Error::msg)
        }?;

        Ok(Self {
            capture,
            resampler,
            buffer: vec![],
            resampled: vec![],
        })
    }

    pub fn capture(&mut self) -> Result<&[f32]> {
        self.capture.capture(&mut self.buffer)?;

        let (n_in, n_out) = self.resample()?;
        _ = self.buffer.drain(..n_in);

        Ok(&self.resampled[..n_out])
    }

    fn resample(&mut self) -> Result<(usize, usize)> {
        self.resampled.clear();

        let n_len = self.buffer.len();
        let mut i_in = 0;
        let mut i_out = 0;

        loop {
            let in_min = self.resampler.input_frames_next();
            if n_len - i_in < in_min {
                break;
            }

            let out_max = self.resampler.output_frames_max();
            self.resampled.resize(i_out + out_max, 0.0);

            let buf = &self.buffer[i_in..];
            let out = &mut self.resampled[i_out..i_out + out_max];

            let (n_in, n_out) = self
                .resampler
                .process_into_buffer(&[buf], &mut [out], None)?;

            i_in += n_in;
            i_out += n_out;
        }

        Ok((i_in, i_out))
    }
}

struct AudioCapture {
    #[allow(unused)]
    audio_device: IMMDevice,

    #[allow(unused)]
    audio_client: IAudioClient,

    capture: IAudioCaptureClient,

    sample_rate: u32,
    n_ch: u32,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        unsafe {
            let device_enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let audio_device: IMMDevice =
                device_enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let audio_client: IAudioClient = audio_device.Activate(CLSCTX_ALL, None)?;

            let (n_ch, sample_rate) = {
                let pwfx = audio_client.GetMixFormat()?;
                let wfx = *pwfx;
                CoTaskMemFree(Some(pwfx as *const _ as _));
                (wfx.nChannels as u32, wfx.nSamplesPerSec)
            };
            let wfx = WAVEFORMATEX {
                wFormatTag: WAVE_FORMAT_IEEE_FLOAT as _,
                nChannels: n_ch as _,
                nSamplesPerSec: sample_rate,
                nAvgBytesPerSec: n_ch * 32 * sample_rate / 8,
                nBlockAlign: n_ch as u16 * 32 / 8,
                wBitsPerSample: 32,
                cbSize: 0,
            };

            let duration = 1000 * 1000 * 10;
            audio_client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK,
                duration,
                0,
                &wfx,
                None,
            )?;

            let capture = audio_client.GetService()?;

            audio_client.Start()?;

            Ok(Self {
                audio_device,
                audio_client,
                capture,
                sample_rate,
                n_ch,
            })
        }
    }

    pub fn capture(&mut self, buf: &mut Vec<f32>) -> Result<()> {
        unsafe {
            loop {
                if self.capture.GetNextPacketSize()? == 0 {
                    break;
                }

                let mut frames: *mut f32 = std::ptr::null_mut();
                let mut n_frames = 0;
                let mut flags = 0;
                self.capture.GetBuffer(
                    &mut frames as *mut _ as _,
                    &mut n_frames,
                    &mut flags,
                    None,
                    None,
                )?;

                let offset = buf.len();
                buf.resize(offset + n_frames as usize, 0.);

                let frames = std::slice::from_raw_parts(frames, (self.n_ch * n_frames) as usize);
                for i in 0..n_frames as usize {
                    for j in 0..self.n_ch as usize {
                        buf[offset + i] += frames[self.n_ch as usize * i + j];
                    }
                }

                self.capture.ReleaseBuffer(n_frames)?;
            }
        }

        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
