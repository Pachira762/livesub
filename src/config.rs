use std::{fmt::Debug, str::FromStr, time::Duration};

use ini::{Ini, SectionSetter};
use windows::Win32::Foundation::RECT;

use crate::gui::utils::Rect as _;

pub const MODEL_SMALL_EN: &str = "distil-whisper/distil-small.en";
pub const MODEL_MEDIUM_EN: &str = "distil-whisper/distil-medium.en";
pub const MODEL_LARGE_V3: &str = "distil-whisper/distil-large-v3";
pub const MODEL_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";
pub const DELAY_LOWEST: Duration = Duration::from_millis(10);
pub const DELAY_LOW: Duration = Duration::from_millis(100);
pub const DELAY_MEDIUM: Duration = Duration::from_millis(300);
pub const DELAY_HIGH: Duration = Duration::from_millis(1000);
pub const DELAY_HIGHEST: Duration = Duration::from_millis(3000);
pub const FONT_NAME_SEGOE_UI: &str = "Segoe UI";
pub const FONT_NAME_ARIAL: &str = "Arial";
pub const FONT_NAME_VERDANA: &str = "Verdana";
pub const FONT_NAME_TAHOMA: &str = "Tahoma";
pub const FONT_NAME_TIMES_NEW_ROMAN: &str = "Times New Roman";
pub const FONT_NAME_CALIBRI: &str = "Calibri";
pub const FONT_SIZE_VERY_SMALL: u32 = 15;
pub const FONT_SIZE_SMALL: u32 = 24;
pub const FONT_SIZE_MEDIUM: u32 = 48;
pub const FONT_SIZE_LARGE: u32 = 64;
pub const FONT_SIZE_VERY_LARGE: u32 = 128;

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub model: String,
    pub latency: Duration,
    pub opacity: f32,
    pub font_name: String,
    pub font_size: u32,
    pub bold: bool,
    pub italic: bool,
    pub outline: bool,
    pub window_rect: RECT,
}

impl Config {
    pub fn load() -> Self {
        let conf = Ini::load_from_file("livesub.ini").unwrap_or_default();
        Self {
            model: conf.get_str("model", MODEL_SMALL_EN),
            latency: Duration::from_millis(conf.get_u32("latency", DELAY_LOW.as_millis() as _) as _),
            opacity: conf.get_u32("opacity", 75) as f32 / 100.0,
            font_name: conf.get_str("font-name", FONT_NAME_SEGOE_UI),
            font_size: conf.get_u32("font-size", FONT_SIZE_SMALL),
            bold: conf.get_bool("font-style-bold", false),
            italic: conf.get_bool("font-style-italic", false),
            outline: conf.get_bool("font-style-outline", false),
            window_rect: RECT::new(
                conf.get_i32("window-x", 100),
                conf.get_i32("window-y", 100),
                conf.get_i32("window-width", 400),
                conf.get_i32("window-height", 200),
            ),
        }
    }

    pub fn save(&self) {
        let mut conf = Ini::new();
        conf.with_general_section()
            .set("model", &self.model)
            .set_u32("latency", self.latency.as_millis() as u32)
            .set_u32("opacity", (100.0 * self.opacity) as _)
            .set("font-name", &self.font_name)
            .set_u32("font-size", self.font_size)
            .set_bool("font-style-bold", self.bold)
            .set_bool("font-style-italic", self.italic)
            .set_bool("font-style-outline", self.outline)
            .set_i32("window-x", self.window_rect.x())
            .set_i32("window-y", self.window_rect.y())
            .set_i32("window-width", self.window_rect.width())
            .set_i32("window-height", self.window_rect.height());

        _ = conf.write_to_file("livesub.ini");
    }
}

trait IniSetter<'a> {
    fn set_bool(&'a mut self, key: &str, value: bool) -> &'a mut SectionSetter<'a>;
    fn set_i32(&'a mut self, key: &str, value: i32) -> &'a mut SectionSetter<'a>;
    fn set_u32(&'a mut self, key: &str, value: u32) -> &'a mut SectionSetter<'a>;
}

impl<'a> IniSetter<'a> for SectionSetter<'a> {
    fn set_bool(&'a mut self, key: &str, value: bool) -> &'a mut SectionSetter<'a> {
        self.set(key, (value as u32).to_string())
    }

    fn set_i32(&'a mut self, key: &str, value: i32) -> &'a mut SectionSetter<'a> {
        self.set(key, value.to_string())
    }

    fn set_u32(&'a mut self, key: &str, value: u32) -> &'a mut SectionSetter<'a> {
        self.set(key, value.to_string())
    }
}

trait IniGetter {
    fn get_bool(&self, key: &str, default: bool) -> bool;
    fn get_i32(&self, key: &str, default: i32) -> i32;
    fn get_u32(&self, key: &str, default: u32) -> u32;
    fn get_str(&self, key: &str, default: &str) -> String;
}

impl IniGetter for Ini {
    fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.general_section().get(key) {
            Some("0") => false,
            Some(_) => true,
            _ => default,
        }
    }

    fn get_i32(&self, key: &str, default: i32) -> i32 {
        i32::from_str(self.general_section().get(key).unwrap_or_default()).unwrap_or(default)
    }

    fn get_u32(&self, key: &str, default: u32) -> u32 {
        u32::from_str(self.general_section().get(key).unwrap_or_default()).unwrap_or(default)
    }

    fn get_str(&self, key: &str, default: &str) -> String {
        self.general_section()
            .get(key)
            .unwrap_or(default)
            .to_string()
    }
}
