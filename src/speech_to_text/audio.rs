use anyhow::Result;
use rubato::{Resampler as _, SincFixedOut, SincInterpolationParameters};
use windows::Win32::{
    Media::{Audio::*, Multimedia::WAVE_FORMAT_IEEE_FLOAT},
    System::Com::*,
};

pub struct Audio {
    raw: Vec<f32>,
    resampled: Vec<f32>,

    capture: AudioCapture,
    resampler: Resampler,
}

impl Audio {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let capture = AudioCapture::new()?;
        let resampler = Resampler::new(capture.sample_rate(), sample_rate)?;

        Ok(Self {
            raw: Vec::new(),
            resampled: Vec::new(),
            capture,
            resampler,
        })
    }

    pub fn capture(&mut self) -> Result<&[f32]> {
        self.capture.capture(&mut self.raw)?;

        self.resampled.clear();
        self.resampler
            .resample(&mut self.raw, &mut self.resampled)?;

        Ok(&self.resampled)
    }

    pub fn clear(&mut self) {
        self.resampled.clear();
        self.raw.clear();
    }
}

struct AudioCapture {
    _audio_device: IMMDevice,
    _audio_client: IAudioClient,
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
                _audio_device: audio_device,
                _audio_client: audio_client,
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

                buf.extend(
                    std::slice::from_raw_parts(frames, (self.n_ch * n_frames) as _)
                        .chunks(self.n_ch as _)
                        .map(|frame| frame.iter().sum::<f32>()),
                );

                self.capture.ReleaseBuffer(n_frames)?;
            }
        }

        Ok(())
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

struct Resampler {
    resampler: SincFixedOut<f32>,
}

impl Resampler {
    fn new(in_sample_rate: u32, out_sample_rate: u32) -> Result<Self> {
        let parameters = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: rubato::SincInterpolationType::Linear,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let resample_ratio = out_sample_rate as f64 / in_sample_rate as f64;
        let resampler = SincFixedOut::<f32>::new(resample_ratio, 8.0, parameters, 1024, 1)?;

        Ok(Self { resampler })
    }

    fn resample(&mut self, input: &mut Vec<f32>, output: &mut Vec<f32>) -> Result<(usize, usize)> {
        let mut i_in = 0;
        let mut i_out = output.len();

        loop {
            let n_next = self.resampler.input_frames_next();
            if input.len() < i_in + n_next {
                break;
            }

            let out_max = self.resampler.output_frames_max();
            output.resize(i_out + out_max, 0.0);

            let wave_in = &input[i_in..i_in + n_next];
            let wave_out = &mut output[i_out..i_out + out_max];

            let (n_in, n_out) =
                self.resampler
                    .process_into_buffer(&[wave_in], &mut [wave_out], None)?;

            i_in += n_in;
            i_out += n_out;
        }

        _ = input.drain(..i_in);
        output.resize(i_out, 0.0);

        Ok((i_in, i_out))
    }
}
