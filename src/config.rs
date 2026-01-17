use std::str::FromStr;

use anyhow::Result;
use ini::Ini;
use windows::Win32::UI::WindowsAndMessaging::CW_USEDEFAULT;

pub const FONT_SIZE_SMALL: i32 = 24;
pub const FONT_SIZE_MEDIUM: i32 = 48;
pub const FONT_SIZE_LARGE: i32 = 96;

#[derive(Debug)]
pub struct Config {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub background: String,
    pub font_name: String,
    pub font_size: i32,
    pub font_style_bold: bool,
    pub font_style_italic: bool,
    pub model_name: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        let conf = Ini::load_from_file("livesub.ini").unwrap_or_default();

        let mut x = CW_USEDEFAULT;
        let mut y = CW_USEDEFAULT;
        let mut width = 640;
        let mut height = 320;
        if let Some(section) = conf.section(Some("window")) {
            x = i32::from_str(section.get("x").unwrap_or_default()).unwrap_or(x);
            y = i32::from_str(section.get("y").unwrap_or_default()).unwrap_or(y);
            width = i32::from_str(section.get("width").unwrap_or_default()).unwrap_or(width);
            height = i32::from_str(section.get("height").unwrap_or_default()).unwrap_or(height);
        }

        let mut font_name = String::from("Arial");
        let mut font_size = FONT_SIZE_MEDIUM;
        let mut font_style_bold = false;
        let mut font_style_italic = false;
        let mut background = String::from("dark");
        if let Some(section) = conf.section(Some("font")) {
            if let Some(name) = section.get("name") {
                font_name = String::from(name);
            }
            font_name = String::from(section.get("name").unwrap_or(&font_name));
            font_size = i32::from_str(section.get("size").unwrap_or_default()).unwrap_or(font_size);
            font_style_bold = bool::from_str(section.get("style-bold").unwrap_or_default())
                .unwrap_or(font_style_bold);
            font_style_italic = bool::from_str(section.get("style-italic").unwrap_or_default())
                .unwrap_or(font_style_italic);
            if let Some(bg) = section.get("background") {
                background = String::from(bg);
            }
        }

        let mut model_name = String::from("parakeet");
        if let Some(section) = conf.section(Some("model"))
            && let Some(name) = section.get("name")
        {
            model_name = String::from(name);
        }

        Ok(Self {
            x,
            y,
            width,
            height,
            background,
            font_name,
            font_size,
            font_style_bold,
            font_style_italic,
            model_name,
        })
    }

    pub fn save(&self) {
        let mut conf = Ini::new();

        conf.with_section(Some("window"))
            .set("x", self.x.to_string())
            .set("y", self.y.to_string())
            .set("width", self.width.to_string())
            .set("height", self.height.to_string());

        conf.with_section(Some("font"))
            .set("name", &self.font_name)
            .set("size", self.font_size.to_string())
            .set("style-bold", self.font_style_bold.to_string())
            .set("style-italic", self.font_style_italic.to_string())
            .set("background", &self.background);

        conf.with_section(Some("model"))
            .set("name", &self.model_name);

        _ = conf.write_to_file("livesub.ini");
    }
}
