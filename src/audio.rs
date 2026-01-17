use anyhow::Result;
use windows::Win32::{
    Media::{
        Audio::{
            AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, IAudioCaptureClient,
            IAudioClient, IMMDevice, IMMDeviceEnumerator, MMDeviceEnumerator, WAVEFORMATEX,
            eConsole, eRender,
        },
        Multimedia::WAVE_FORMAT_IEEE_FLOAT,
    },
    System::Com::{CLSCTX_ALL, CoCreateInstance, CoTaskMemFree},
};

pub struct AudioSource {
    _audio_device: IMMDevice,
    _audio_client: IAudioClient,
    capture: IAudioCaptureClient,
    sample_rate: u32,
    n_channels: u32,
    samples: Vec<f32>,
}

impl AudioSource {
    pub fn new() -> Result<Self> {
        unsafe {
            let device_enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;
            let audio_device: IMMDevice =
                device_enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
            let audio_client: IAudioClient = audio_device.Activate(CLSCTX_ALL, None)?;

            let (n_channels, sample_rate) = {
                let pwfx = audio_client.GetMixFormat()?;
                let wfx = *pwfx;
                CoTaskMemFree(Some(pwfx as *const _ as _));
                (wfx.nChannels as u32, wfx.nSamplesPerSec)
            };

            let wfx = WAVEFORMATEX {
                wFormatTag: WAVE_FORMAT_IEEE_FLOAT as _,
                nChannels: n_channels as _,
                nSamplesPerSec: sample_rate,
                nAvgBytesPerSec: n_channels * 32 * sample_rate / 8,
                nBlockAlign: n_channels as u16 * 32 / 8,
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

            let capture: IAudioCaptureClient = audio_client.GetService()?;
            audio_client.Start()?;

            Ok(Self {
                _audio_device: audio_device,
                _audio_client: audio_client,
                capture,
                sample_rate,
                n_channels,
                samples: vec![],
            })
        }
    }

    pub fn capture(&mut self) -> Result<usize> {
        unsafe {
            self.samples.clear();

            while self.capture.GetNextPacketSize()? > 0 {
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

                self.samples.reserve(n_frames as usize);
                self.samples.extend(
                    std::slice::from_raw_parts(frames, (self.n_channels * n_frames) as _)
                        .chunks(self.n_channels as _)
                        .map(|channels| channels.iter().sum::<f32>()),
                );

                self.capture.ReleaseBuffer(n_frames)?;
            }

            Ok(self.samples.len())
        }
    }

    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
