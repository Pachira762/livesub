use std::{
    sync::mpsc::{Receiver, Sender},
    thread::JoinHandle,
    time::Duration,
};

use anyhow::Result;
use audio::Audio;
use candle_transformers::models::whisper::SAMPLE_RATE;
use text::TextStream;
use transcribe::Transcriber;
use windows::Win32::{
    System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED},
    UI::WindowsAndMessaging::{MessageBoxA, MB_OK},
};
use windows_core::{s, PCSTR};

mod audio;
mod mel;
mod text;
mod transcribe;

pub struct SpeechToText {
    sender: Sender<Message>,
    handle: Option<JoinHandle<Result<()>>>,
    ts: TextStream,
}

impl SpeechToText {
    pub fn new(repo_id: &str, latency: Duration) -> Result<Self> {
        let ts = TextStream::new();
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut ctx = SpeechToTextContext::new(latency, ts.clone(), receiver)?;
        let handle = Some(std::thread::spawn(move || -> Result<()> {
            unsafe { RoInitialize(RO_INIT_MULTITHREADED) }?;
            ctx.process()
        }));

        _ = sender.send(Message::Model(repo_id.to_string()));

        Ok(Self { sender, handle, ts })
    }

    pub fn text(&mut self) -> Option<String> {
        self.ts.get()
    }

    pub fn set_model(&self, repo_id: &str) {
        _ = self.sender.send(Message::Model(repo_id.to_string()));
    }

    pub fn set_latency(&self, latency: Duration) {
        _ = self.sender.send(Message::Latency(latency.as_millis() as _));
    }

    pub fn clear(&self) {
        _ = self.sender.send(Message::Claer);
    }
}

impl Drop for SpeechToText {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            _ = self.sender.send(Message::Quit);

            if let Ok(Err(e)) = handle.join() {
                unsafe {
                    let text = format!("{e:?}\0");
                    MessageBoxA(None, PCSTR(text.as_ptr()), s!("error"), MB_OK);
                }
            }
        }
    }
}

struct SpeechToTextContext {
    audio: Audio,
    transcriber: Option<Transcriber>,
    ts: TextStream,
    latency: Duration,
    receiver: Receiver<Message>,
    keep_running: bool,
}

impl SpeechToTextContext {
    fn new(latency: Duration, ts: TextStream, receiver: Receiver<Message>) -> Result<Self> {
        let audio = Audio::new(SAMPLE_RATE as _)?;

        Ok(Self {
            audio,
            transcriber: None,
            ts,
            latency,
            receiver,
            keep_running: true,
        })
    }

    fn process(&mut self) -> Result<()> {
        while self.keep_running {
            if self.recieve_message()? {
                continue;
            }

            if self.transcriber.is_some() {
                self.transcribe()?;
            }
        }

        Ok(())
    }

    fn recieve_message(&mut self) -> Result<bool> {
        if let Ok(message) = self.receiver.recv_timeout(self.latency) {
            match message {
                Message::Quit => {
                    self.keep_running = false;
                }
                Message::Claer => {
                    if let Some(transcriber) = &mut self.transcriber {
                        transcriber.clear();
                    }
                    self.audio.clear();
                    self.ts.clear();
                }
                Message::Model(repo_id) => {
                    self.ts.clear();
                    self.ts.set(format!("Loading {repo_id}\r\n"), true);

                    match Transcriber::new(&repo_id) {
                        Ok(transcriber) => {
                            self.ts.clear();
                            self.transcriber = Some(transcriber)
                        }
                        Err(e) => {
                            self.ts.set(format!("{e:?}"), true);
                        }
                    }
                }
                Message::Latency(latency) => {
                    self.latency = Duration::from_millis(latency as _);
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn transcribe(&mut self) -> Result<()> {
        let audio = self.audio.capture()?;

        let result = if let Some(transcruber) = &mut self.transcriber {
            transcruber.transcribe(audio)?
        } else {
            None
        };

        if let Some((text, is_new_segment)) = result {
            self.ts.set(text, is_new_segment);
        }

        Ok(())
    }
}

unsafe impl Send for SpeechToTextContext {}

enum Message {
    Quit,
    Claer,
    Model(String),
    Latency(u32),
}
