use std::{fmt::Debug, str::FromStr, time::Duration};

use ini::{Ini, SectionSetter};
use strum_macros::{AsRefStr, EnumString};

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, EnumString, AsRefStr)]
pub enum Model {
    #[default]
    #[strum(serialize = "distil-whisper/distil-small.en")]
    DistilSmallEn,
    #[strum(serialize = "distil-whisper/distil-medium.en")]
    DistilMediumEn,
    #[strum(serialize = "distil-whisper/distil-large-v3")]
    DistilLargeV3,
    #[strum(serialize = "openai/whisper-large-v3")]
    WhisperLargeV3,
}

impl Model {
    pub fn repo_id(&self) -> &str {
        self.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, EnumString, AsRefStr)]
pub enum Delay {
    #[strum(serialize = "none")]
    None,
    #[default]
    #[strum(serialize = "low")]
    Low,
    #[strum(serialize = "mid")]
    Mid,
    #[strum(serialize = "high")]
    High,
    #[strum(serialize = "highest")]
    Highest,
}

impl Delay {
    pub fn duration(&self) -> Duration {
        match *self {
            Delay::None => Duration::from_millis(1),
            Delay::Low => Duration::from_millis(100),
            Delay::Mid => Duration::from_millis(300),
            Delay::High => Duration::from_millis(500),
            Delay::Highest => Duration::from_millis(3000),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, EnumString, AsRefStr)]
pub enum Transparency {
    #[strum(serialize = "none")]
    None,
    #[strum(serialize = "low")]
    Low,
    #[default]
    #[strum(serialize = "mid")]
    Mid,
    #[strum(serialize = "high")]
    High,
    #[strum(serialize = "transparent")]
    Transparent,
}

impl Transparency {
    pub fn alph(&self) -> f32 {
        match *self {
            Transparency::None => 1.0,
            Transparency::Low => 0.8,
            Transparency::Mid => 0.5,
            Transparency::High => 0.2,
            Transparency::Transparent => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, EnumString, AsRefStr)]
pub enum FontFamily {
    // Sans-serif
    #[strum(serialize = "Aptos")]
    Aptos,
    #[default]
    #[strum(serialize = "Arial")]
    Arial,
    #[strum(serialize = "Calibri")]
    Calibri,
    #[strum(serialize = "Segoe UI")]
    SegoeUI,

    // Sans
    #[strum(serialize = "Cambria")]
    Cambria,
    #[strum(serialize = "Georgia")]
    Georgia,

    // Other
    #[strum(serialize = "Consola")]
    Consola,
    #[strum(serialize = "Comic Sans MS")]
    ComicSansMS,
    #[strum(serialize = "Impact")]
    Impact,
}

impl FontFamily {
    pub fn name(&self) -> &str {
        self.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, EnumString, AsRefStr)]
pub enum FontSize {
    #[strum(serialize = "very-small")]
    VerySmall,
    #[default]
    #[strum(serialize = "small")]
    Small,
    #[strum(serialize = "medium")]
    Medium,
    #[strum(serialize = "large")]
    Large,
    #[strum(serialize = "very-large")]
    VeryLarge,
}

impl FontSize {
    pub fn size(&self) -> f32 {
        match *self {
            FontSize::VerySmall => 18.0,
            FontSize::Small => 24.0,
            FontSize::Medium => 32.0,
            FontSize::Large => 48.0,
            FontSize::VeryLarge => 64.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    pub model: Model,
    pub delay: Delay,
    pub transparency: Transparency,
    pub font_family: FontFamily,
    pub font_size: FontSize,
    pub bold: bool,
    pub italic: bool,
    pub outline: bool,
    pub width: u32,
    pub height: u32,
}

impl Config {
    pub fn save(&self) {
        let mut conf = Ini::new();
        conf.with_general_section()
            .set_enum("model", &self.model)
            .set_enum("delay", &self.delay)
            .set_enum("transparency", &self.transparency)
            .set_enum("font-family", &self.font_family)
            .set_enum("font-size", &self.font_size)
            .set_bool("bold", self.bold)
            .set_bool("italic", self.italic)
            .set_bool("outline", self.outline)
            .set("window-width", self.width.to_string())
            .set("window-height", self.height.to_string());

        _ = conf.write_to_file("livesub.ini");
    }

    pub fn from_ini() -> Self {
        if let Ok(conf) = Ini::load_from_file_noescape("livesub.ini") {
            let model: Model = conf.get_enum("model");
            let delay: Delay = conf.get_enum("delay");
            let transparency: Transparency = conf.get_enum("transparency");
            let font_family: FontFamily = conf.get_enum("font-family");
            let font_size: FontSize = conf.get_enum("font-size");
            let bold = conf.get_bool("bold");
            let italic = conf.get_bool("italic");
            let outline = conf.get_bool("outline");
            let width = conf.get_u32("window-width", 640);
            let height = conf.get_u32("window-height", 160);

            Self {
                model,
                delay,
                transparency,
                font_family,
                font_size,
                bold,
                italic,
                outline,
                width,
                height,
            }
        } else {
            Self {
                width: 640,
                height: 160,
                ..Default::default()
            }
        }
    }
}

trait IniSetter<'a> {
    fn set_bool(&'a mut self, key: &str, value: bool) -> &'a mut SectionSetter<'a>;
    fn set_enum<T: AsRef<str>>(&'a mut self, key: &str, value: &T) -> &'a mut SectionSetter<'a>;
}

impl<'a> IniSetter<'a> for SectionSetter<'a> {
    fn set_bool(&'a mut self, key: &str, value: bool) -> &'a mut SectionSetter<'a> {
        self.set(key, (value as u32).to_string())
    }

    fn set_enum<T: AsRef<str>>(&'a mut self, key: &str, value: &T) -> &'a mut SectionSetter<'a> {
        self.set(key, value.as_ref())
    }
}

trait IniGetter {
    fn get_bool(&self, key: &str) -> bool;
    fn get_u32(&self, key: &str, default: u32) -> u32;
    fn get_enum<T: Default + FromStr>(&self, key: &str) -> T;
}

impl IniGetter for Ini {
    fn get_bool(&self, key: &str) -> bool {
        match self.get_from::<String>(None, key) {
            Some("0") => false,
            Some("1") => true,
            _ => false,
        }
    }

    fn get_u32(&self, key: &str, default: u32) -> u32 {
        u32::from_str(self.get_from::<String>(None, key).unwrap_or_default()).unwrap_or(default)
    }

    fn get_enum<T: Default + FromStr>(&self, key: &str) -> T {
        T::from_str(self.get_from::<String>(None, key).unwrap_or_default()).unwrap_or_default()
    }
}
