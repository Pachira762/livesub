mod command;
mod config;
pub use config::Config;

use std::{
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::Result;
use windows::Win32::{
    Foundation::HWND,
    System::Com::*,
    UI::WindowsAndMessaging::{DestroyWindow, SetTimer},
};

use crate::{
    gui::{MenuBuilder, WinApp},
    render::Renderer,
    transcribe::{TextStream, TranscribeResult, Transcriber},
};

use command::{Command, Command::*};
use config::*;

pub struct App {
    config: Config,
    hwnd: HWND,
    renderer: Renderer,
    tx: Sender<Command>,
    text_stream: Arc<Mutex<TextStream>>,
    handle: Option<JoinHandle<()>>,
}

impl WinApp for App {
    type Command = Command;

    fn new(config: Config, hwnd: HWND, menu: &mut MenuBuilder) -> Result<Self> {
        let font_name = config.font_family.name();
        let font_size = config.font_size.size();
        let transparency = config.transparency.alph();
        let renderer = Renderer::new(hwnd, font_name, font_size, transparency)?;
        let (tx, rx) = mpsc::channel::<Command>();
        let text_stream = Arc::new(Mutex::new(TextStream::new()));
        let handle = {
            let text_stream = Arc::clone(&text_stream);
            let duration = config.delay.duration();
            Some(thread::spawn(move || {
                transcribe_proc(rx, text_stream, duration)
            }))
        };

        menu.add_cmd(Clear, "Clear")?
            .separate()?
            .push_submenu("Model")?
            .add_radio_group()?
            .add_radio(
                ModelDistilSmallEn,
                "distil-small.en",
                config.model == Model::DistilSmallEn,
            )?
            .add_radio(
                ModelDistilMediumEn,
                "distil-medium.en",
                config.model == Model::DistilMediumEn,
            )?
            .add_radio(
                ModelDistilLargeV2,
                "distil-large-v2",
                config.model == Model::DistilLargeV2,
            )?
            // .add_radio(
            //     ModelWhisperLargeV3,
            //     "whisper-large-v3",
            //     config.model == Model::WhisperLargeV3,
            // )?
            .pop_submenu()?
            .push_submenu("Delay")?
            .add_radio_group()?
            .add_radio(DelayNone, "None", config.delay == Delay::None)?
            .add_radio(DelayLow, "Low", config.delay == Delay::Low)?
            .add_radio(DelayMid, "Midium", config.delay == Delay::Mid)?
            .add_radio(DelayHigh, "High", config.delay == Delay::High)?
            .add_radio(DelayHighest, "Highest", config.delay == Delay::Highest)?
            .pop_submenu()?
            .push_submenu("Transparency")?
            .add_radio_group()?
            .add_radio(
                TransparencyNone,
                "None",
                config.transparency == Transparency::None,
            )?
            .add_radio(
                TransparencyLow,
                "Low",
                config.transparency == Transparency::Low,
            )?
            .add_radio(
                TransparencyMid,
                "Mid",
                config.transparency == Transparency::Mid,
            )?
            .add_radio(
                TransparencyHigh,
                "High",
                config.transparency == Transparency::High,
            )?
            .add_radio(
                Transparent,
                "Transparent",
                config.transparency == Transparency::Transparent,
            )?
            .pop_submenu()?
            .push_submenu("Font Family")?
            .add_radio_group()?
            .add_radio(
                FontFamilyAptos,
                "Aptos",
                config.font_family == FontFamily::Aptos,
            )?
            .add_radio(
                FontFamilyArial,
                "Arial",
                config.font_family == FontFamily::Arial,
            )?
            .add_radio(
                FontFamilyCalibri,
                "Calibri",
                config.font_family == FontFamily::Calibri,
            )?
            .add_radio(
                FontFamilySegoeUI,
                "SegoeUI",
                config.font_family == FontFamily::SegoeUI,
            )?
            .add_radio(
                FontFamilyCambria,
                "Cambria",
                config.font_family == FontFamily::Cambria,
            )?
            .add_radio(
                FontFamilyGeorgia,
                "Georgia",
                config.font_family == FontFamily::Georgia,
            )?
            .add_radio(
                FontFamilyConsola,
                "Consola",
                config.font_family == FontFamily::Consola,
            )?
            .add_radio(
                FontFamilyComicSansMS,
                "Comic Sans",
                config.font_family == FontFamily::ComicSansMS,
            )?
            .add_radio(
                FontFamilyImpact,
                "Impact",
                config.font_family == FontFamily::Impact,
            )?
            .pop_submenu()?
            .push_submenu("Text Size")?
            .add_radio_group()?
            .add_radio(
                FontSizeVerySmall,
                "Very Small",
                config.font_size == FontSize::VerySmall,
            )?
            .add_radio(FontSizeSmall, "Small", config.font_size == FontSize::Small)?
            .add_radio(
                FontSizeMedium,
                "Medium",
                config.font_size == FontSize::Medium,
            )?
            .add_radio(FontSizeLarge, "Large", config.font_size == FontSize::Large)?
            .add_radio(
                FontSizeVeryLarge,
                "Very Large",
                config.font_size == FontSize::VeryLarge,
            )?
            .pop_submenu()?
            .push_submenu("Text Style")?
            .add_checkbox(TextStyleBold, "Bold", config.bold)?
            .add_checkbox(TextStyleItalic, "Italic", config.italic)?
            .add_checkbox(TextStyleOutline, "Outline", config.outline)?
            .pop_submenu()?
            .separate()?
            .add_cmd(Quit, "Quit(&Q)")?;

        unsafe {
            let duration = config.delay.duration().as_millis() / 2;
            SetTimer(hwnd, 1, duration as u32, None);
        }

        Ok(Self {
            config,
            hwnd,
            renderer,
            tx,
            text_stream,
            handle,
        })
    }

    fn on_close(&mut self) {
        self.config.save()
    }

    fn on_sized(&mut self, cx: i32, cy: i32) {
        self.config.width = cx as u32;
        self.config.height = cy as u32;
        self.renderer.resize(cx as _, cy as _);
    }

    fn on_paint(&mut self) {
        self.renderer.draw();
    }

    fn on_timer(&mut self) {
        if let Ok(mut text_stream) = self.text_stream.lock() {
            if let Some(text) = text_stream.get() {
                self.renderer.set_text(&text);
                self.renderer.draw();
            }
        }
    }

    fn on_menu(&mut self, id: Command, state: Option<bool>) {
        match id {
            Quit => unsafe {
                self.on_close();
                _ = DestroyWindow(self.hwnd);
            },
            ModelDistilSmallEn | ModelDistilMediumEn | ModelDistilLargeV2 | ModelWhisperLargeV3 => {
                self.config.model = id.into();
            }
            DelayNone | DelayLow | DelayMid | DelayHigh | DelayHighest => {
                self.config.delay = id.into();
                let duration = self.config.delay.duration().as_millis() / 2;
                unsafe {
                    SetTimer(self.hwnd, 1, duration as u32, None);
                }
            }
            TransparencyNone | TransparencyLow | TransparencyMid | TransparencyHigh
            | Transparent => {
                self.config.transparency = id.into();
                self.renderer
                    .set_transparency(self.config.transparency.alph());
                self.renderer.draw();
            }
            FontFamilyAptos
            | FontFamilyArial
            | FontFamilyCalibri
            | FontFamilySegoeUI
            | FontFamilyCambria
            | FontFamilyGeorgia
            | FontFamilyConsola
            | FontFamilyComicSansMS
            | FontFamilyImpact => {
                self.config.font_family = id.into();
                self.renderer.set_font_name(self.config.font_family.name());
                self.renderer.draw();
            }
            FontSizeVerySmall | FontSizeSmall | FontSizeMedium | FontSizeLarge
            | FontSizeVeryLarge => {
                self.config.font_size = id.into();
                self.renderer.set_font_size(self.config.font_size.size());
                self.renderer.draw();
            }
            TextStyleBold => {
                self.renderer.set_bold(state.unwrap_or_default());
                self.renderer.draw();
            }
            TextStyleItalic => {
                self.renderer.set_italic(state.unwrap_or_default());
                self.renderer.draw();
            }
            TextStyleOutline => {
                self.renderer.set_outline(state.unwrap_or_default());
                self.renderer.draw();
            }
            _ => {}
        }

        _ = self.tx.send(id);
    }

    fn on_dpi_changed(&mut self, dpi: u32) {
        self.renderer.set_dpi(dpi);
    }
}

impl Drop for App {
    fn drop(&mut self) {
        _ = self.tx.send(Quit);

        if let Some(handle) = self.handle.take() {
            _ = handle.join();
        }
    }
}

fn transcribe_proc(rx: Receiver<Command>, text_stream: Arc<Mutex<TextStream>>, duration: Duration) {
    unsafe {
        _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        _ = transcribe_proc_(rx, text_stream, duration);
        CoUninitialize();
    }
}

fn transcribe_proc_(
    rx: Receiver<Command>,
    text_stream: Arc<Mutex<TextStream>>,
    mut duration: Duration,
) -> Result<()> {
    set_text(&text_stream, "Loading Model...");
    let mut transcriber = Transcriber::new("distil-whisper/distil-small.en")?;
    clear_text(&text_stream);

    loop {
        if let Ok(id) = rx.recv_timeout(duration) {
            match id {
                Quit => {
                    break;
                }
                Clear => {
                    transcriber.clear();
                    clear_text(&text_stream);
                }
                ModelDistilSmallEn | ModelDistilMediumEn | ModelDistilLargeV2
                | ModelWhisperLargeV3 => {
                    let model: Model = id.into();
                    update_model(&mut transcriber, model.repo_id(), &text_stream)?;
                }
                DelayNone | DelayLow | DelayMid | DelayHigh | DelayHighest => {
                    let delay: Delay = id.into();
                    duration = delay.duration();
                }
                _ => {}
            }
        }

        if let Ok(Some(TranscribeResult { text, new_segment })) = transcriber.transcribe() {
            if let Ok(mut text_stream) = text_stream.lock() {
                text_stream.set(text, new_segment);
            }
        }
    }

    Ok(())
}

fn update_model(
    transcriber: &mut Transcriber,
    repo_id: &str,
    text_stream: &Arc<Mutex<TextStream>>,
) -> Result<()> {
    set_text(text_stream, "Loading Model...");
    transcriber.set_model(repo_id)?;
    clear_text(text_stream);

    Ok(())
}

fn set_text(text_stream: &Arc<Mutex<TextStream>>, new_text: &str) {
    if let Ok(mut ts) = text_stream.lock() {
        ts.set_force(new_text.to_owned());
    }
}

fn clear_text(text_stream: &Arc<Mutex<TextStream>>) {
    if let Ok(mut ts) = text_stream.lock() {
        ts.clear();
    }
}
