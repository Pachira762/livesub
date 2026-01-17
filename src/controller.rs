use std::{
    sync::{Arc, Mutex, mpsc::Sender},
    thread::JoinHandle,
};

use anyhow::Result;
use windows::Win32::{Foundation::HWND, UI::WindowsAndMessaging::WM_USER};

pub const WM_NEW_TRANSCRIPTION: u32 = WM_USER + 2;

pub struct Controller {
    sender: Sender<Message>,
    joiner: Option<JoinHandle<Result<()>>>,
}

impl Controller {
    pub fn new(hwnd: HWND, text: Arc<Mutex<String>>, model_name: &str) -> Result<Self> {
        let (sender, receiver) = std::sync::mpsc::channel();
        let joiner = Some(worker::Worker::spawn(
            hwnd,
            text,
            receiver,
            model_name.into(),
        ));
        Ok(Self { sender, joiner })
    }

    pub fn clear(&self) -> Result<()> {
        self.sender.send(Message::Clear)?;
        Ok(())
    }

    pub fn set_model(&self, model_name: &str) -> Result<()> {
        self.sender.send(Message::Model(model_name.into()))?;
        Ok(())
    }

    pub fn shutdown(&mut self) {
        if let Some(joiner) = self.joiner.take() {
            _ = self.sender.send(Message::Quit);
            match joiner.join() {
                Ok(ret) => {
                    if let Err(e) = ret {
                        eprintln!("{e:?}");
                    }
                }
                Err(e) => {
                    eprintln!("{e:?}");
                }
            }
        }
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(PartialEq)]
enum Message {
    Quit,
    Clear,
    Model(String),
}

mod worker {
    use std::{
        sync::{
            Arc, Mutex,
            mpsc::{Receiver, TryRecvError},
        },
        thread::JoinHandle,
    };

    use anyhow::Result;
    use windows::Win32::{
        Foundation::{HWND, LPARAM, WPARAM},
        System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize},
        UI::WindowsAndMessaging::{PostMessageA, WM_CLOSE},
    };

    use crate::{
        asr::{self, transcribe::Transcriber},
        audio::AudioSource,
        controller::{Message, WM_NEW_TRANSCRIPTION},
    };

    pub struct Worker {
        running: bool,
        hwnd: HWND,
        receiver: Receiver<Message>,
        audio_source: AudioSource,
        transcriber: Transcriber,
        text: Arc<Mutex<String>>,
    }

    impl Worker {
        pub fn spawn(
            hwnd: HWND,
            text: Arc<Mutex<String>>,
            receiver: Receiver<Message>,
            model_name: String,
        ) -> JoinHandle<Result<()>> {
            let hwnd = hwnd.0 as usize;
            std::thread::spawn(move || {
                let hwnd = HWND(hwnd as _);
                if let Err(e) = Self::run(hwnd, text, receiver, model_name) {
                    unsafe {
                        _ = PostMessageA(
                            Some(hwnd),
                            WM_CLOSE,
                            WPARAM::default(),
                            LPARAM::default(),
                        );
                    }
                    Err(e)
                } else {
                    Ok(())
                }
            })
        }

        fn run(
            hwnd: HWND,
            text: Arc<Mutex<String>>,
            receiver: Receiver<Message>,
            model_name: String,
        ) -> Result<()> {
            unsafe {
                CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;

                let audio_source = AudioSource::new()?;
                let transcriber = Transcriber::new(audio_source.sample_rate(), &model_name)?;
                let mut worker = Self {
                    running: true,
                    hwnd,
                    receiver,
                    audio_source,
                    transcriber,
                    text,
                };

                while worker.running {
                    match worker.receiver.try_recv() {
                        Ok(message) => {
                            worker.handle_message(message)?;
                        }
                        Err(TryRecvError::Empty) => {
                            worker.work()?;
                        }
                        Err(e) => anyhow::bail!(e),
                    }
                }

                asr::shutdow();
                CoUninitialize();
                Ok(())
            }
        }

        fn handle_message(&mut self, message: Message) -> Result<()> {
            match message {
                Message::Quit => {
                    self.running = false;
                }
                Message::Clear => {
                    self.transcriber.reset()?;
                }
                Message::Model(model_name) => {
                    self.transcriber.set_model(&model_name)?;
                }
            }
            Ok(())
        }

        fn work(&mut self) -> Result<()> {
            if self.audio_source.capture()? == 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                return Ok(());
            }

            let samples = self.audio_source.samples();
            let (confirmed, tentative) = self.transcriber.transcribe(samples)?;

            if let Some(text) = confirmed {
                self.notify_text(text)?;
            } else if let Some(text) = tentative {
                self.notify_text(text)?;
            }

            std::thread::sleep(std::time::Duration::from_millis(10));
            Ok(())
        }

        fn notify_text(&mut self, text: String) -> Result<()> {
            unsafe {
                if let Ok(mut lock) = self.text.lock() {
                    *lock = text;

                    PostMessageA(
                        Some(self.hwnd),
                        WM_NEW_TRANSCRIPTION,
                        WPARAM::default(),
                        LPARAM::default(),
                    )?;
                }
                Ok(())
            }
        }
    }
}
