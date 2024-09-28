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
use windows::Win32::System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED};

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
        let mut ctx = SpeechToTextContext::new(repo_id, latency, ts.clone(), receiver)?;
        let handle = Some(std::thread::spawn(move || -> Result<()> {
            unsafe { RoInitialize(RO_INIT_MULTITHREADED) }?;
            ctx.process()
        }));

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
                println!("{e:?}");
            }
        }
    }
}

struct SpeechToTextContext {
    audio: Audio,
    transcriber: Transcriber,
    ts: TextStream,
    latency: Duration,
    receiver: Receiver<Message>,
    keep_running: bool,
}

impl SpeechToTextContext {
    fn new(
        repo_id: &str,
        latency: Duration,
        ts: TextStream,
        receiver: Receiver<Message>,
    ) -> Result<Self> {
        let audio = Audio::new(SAMPLE_RATE as _)?;
        let transcriber = Transcriber::new(repo_id)?;

        Ok(Self {
            audio,
            transcriber,
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

            self.transcribe()?;
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
                    self.transcriber.clear();
                    self.audio.clear();
                    self.ts.clear();
                }
                Message::Model(repo_id) => {
                    self.transcriber = Transcriber::new(&repo_id)?;
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
        if let Some((text, is_new_segment)) = self.transcriber.transcribe(audio)? {
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
