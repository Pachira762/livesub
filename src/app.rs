use std::time::Duration;

use anyhow::Result;
use windows::Win32::Foundation::HWND;

use crate::{
    action, checkbox,
    config::*,
    graphics::Renderer,
    gui::{
        app::{App as GuiApp, MenuItem},
        utils::Hwnd as _,
    },
    radio, separator,
    speech_to_text::SpeechToText,
    submenu,
};

const TIMER_ID: usize = 0x01;

pub struct App {
    config: Config,
    hwnd: HWND,
    s2t: SpeechToText,
    renderer: Renderer,
}

impl App {
    fn clear(&mut self) {
        self.s2t.clear();
        self.renderer.set_text("");
    }

    fn set_model(&mut self, repo_id: &str) {
        self.config.model = repo_id.into();
        self.s2t.set_model(repo_id);
    }

    fn set_latency(&mut self, latency: Duration) {
        self.config.latency = latency;
        self.s2t.set_latency(self.config.latency);
        self.hwnd
            .set_timer(TIMER_ID, latency.as_millis() as u32 / 2);
    }

    fn set_opacity(&mut self, opacity: f32) {
        self.config.opacity = opacity;
        self.renderer.set_opacity(opacity);
    }

    fn set_font_name(&mut self, font_name: &str) {
        self.config.font_name = font_name.into();
        self.renderer.set_font_name(font_name);
    }

    fn set_font_size(&mut self, font_size: u32) {
        self.config.font_size = font_size;
        self.renderer.set_font_size(font_size);
    }

    fn set_font_style_bold(&mut self, bold: bool) {
        self.config.bold = bold;
        self.renderer.set_bold(bold);
    }

    fn set_font_style_italic(&mut self, italic: bool) {
        self.config.italic = italic;
        self.renderer.set_italic(italic);
    }

    fn set_font_style_outline(&mut self, outline: bool) {
        self.config.outline = outline;
        self.renderer.set_outline(outline);
    }

    fn quit(&mut self) {
        self.hwnd.destroy();
    }
}

impl GuiApp for App {
    fn new(config: Config, hwnd: HWND) -> Result<Self> {
        let s2t = SpeechToText::new(&config.model, config.latency)?;

        let renderer = Renderer::new(
            hwnd,
            &config.font_name,
            config.font_size,
            config.bold,
            config.italic,
            config.outline,
            config.opacity,
        )?;

        _ = hwnd.set_timer(TIMER_ID, config.latency.as_millis() as u32 / 2);

        Ok(Self {
            config,
            hwnd,
            s2t,
            renderer,
        })
    }

    fn on_close(&mut self) {
        self.config.save()
    }

    fn on_move(&mut self, _x: i32, _y: i32) {
        self.config.window_rect = self.hwnd.rect();
    }

    fn on_sized(&mut self, cx: i32, cy: i32) {
        if cx > 0 && cy > 0 {
            self.config.window_rect = self.hwnd.rect();
            _ = self.renderer.set_size(cx as _, cy as _);
        }
    }

    fn on_paint(&mut self) {
        _ = self.renderer.draw();
    }

    fn on_timer(&mut self) {
        if let Some(text) = self.s2t.text() {
            self.renderer.set_text(&text);
        }
    }

    fn on_dpi_changed(&mut self, dpi: u32) {
        self.renderer.set_dpi(dpi);
    }

    fn on_menu(&mut self, id: u32, state: bool) {
        match id {
            CMD_CLEAR => self.clear(),
            CMD_MODEL_SMALL_EN => self.set_model(MODEL_SMALL_EN),
            CMD_MODEL_MEDIUM_EN => self.set_model(MODEL_MEDIUM_EN),
            CMD_MODEL_LARGE_V3 => self.set_model(MODEL_LARGE_V3),
            CMD_DELAY_LOWEST => self.set_latency(DELAY_LOWEST),
            CMD_DELAY_LOW => self.set_latency(DELAY_LOW),
            CMD_DELAY_MEDIUM => self.set_latency(DELAY_MEDIUM),
            CMD_DELAY_HIGH => self.set_latency(DELAY_HIGH),
            CMD_DELAY_HIGHEST => self.set_latency(DELAY_HIGHEST),
            CMD_TRANSPARENCY_0 => self.set_opacity(0.0),
            CMD_TRANSPARENCY_25 => self.set_opacity(0.25),
            CMD_TRANSPARENCY_50 => self.set_opacity(0.5),
            CMD_TRANSPARENCY_75 => self.set_opacity(0.75),
            CMD_TRANSPARENCY_100 => self.set_opacity(1.0),
            CMD_FONT_NAME_SEGOE_UI => self.set_font_name(FONT_NAME_SEGOE_UI),
            CMD_FONT_NAME_ARIAL => self.set_font_name(FONT_NAME_ARIAL),
            CMD_FONT_NAME_VERDANA => self.set_font_name(FONT_NAME_VERDANA),
            CMD_FONT_NAME_TAHOMA => self.set_font_name(FONT_NAME_TAHOMA),
            CMD_FONT_NAME_TIMES_NEW_ROMAN => self.set_font_name(FONT_NAME_TIMES_NEW_ROMAN),
            CMD_FONT_NAME_CALIBRI => self.set_font_name(FONT_NAME_CALIBRI),
            CMD_FONT_SIZE_VERY_SMALL => self.set_font_size(FONT_SIZE_VERY_SMALL),
            CMD_FONT_SIZE_SMALL => self.set_font_size(FONT_SIZE_SMALL),
            CMD_FONT_SIZE_MEDIUM => self.set_font_size(FONT_SIZE_MEDIUM),
            CMD_FONT_SIZE_LARGE => self.set_font_size(FONT_SIZE_LARGE),
            CMD_FONT_SIZE_VERY_LARGE => self.set_font_size(FONT_SIZE_VERY_LARGE),
            CMD_FONT_STYLE_BOLD => self.set_font_style_bold(state),
            CMD_FONT_STYLE_ITALIC => self.set_font_style_italic(state),
            CMD_FONT_STYLE_OUTLINE => self.set_font_style_outline(state),
            CMD_QUIT => self.quit(),
            _ => {}
        }
    }

    fn menu_items(&self) -> Vec<MenuItem> {
        let config = &self.config;

        vec![
            action!(CMD_CLEAR, "Clear"),
            separator!(),
            submenu!(
                "Model",
                radio!(
                    CMD_MODEL_SMALL_EN,
                    "distil-small.en",
                    config.model == MODEL_SMALL_EN,
                ),
                radio!(
                    CMD_MODEL_MEDIUM_EN,
                    "distil-medium.en",
                    config.model == MODEL_MEDIUM_EN,
                ),
                radio!(
                    CMD_MODEL_LARGE_V3,
                    "distil-large-v3",
                    config.model == MODEL_LARGE_V3,
                ),
            ),
            submenu!(
                "Latency",
                radio!(CMD_DELAY_LOWEST, "Lowest", config.latency == DELAY_LOWEST),
                radio!(CMD_DELAY_LOW, "Low", config.latency == DELAY_LOW),
                radio!(CMD_DELAY_MEDIUM, "Medium", config.latency == DELAY_MEDIUM),
                radio!(CMD_DELAY_HIGH, "High", config.latency == DELAY_HIGH),
                radio!(
                    CMD_DELAY_HIGHEST,
                    "Highest",
                    config.latency == DELAY_HIGHEST
                ),
            ),
            submenu!(
                "Opacity",
                radio!(CMD_TRANSPARENCY_0, "0%", config.opacity == 0.0),
                radio!(CMD_TRANSPARENCY_25, "25%", config.opacity == 0.25),
                radio!(CMD_TRANSPARENCY_50, "50%", config.opacity == 0.5),
                radio!(CMD_TRANSPARENCY_75, "75%", config.opacity == 0.75),
                radio!(CMD_TRANSPARENCY_100, "100%", config.opacity == 1.0),
            ),
            submenu!(
                "Font",
                radio!(
                    CMD_FONT_NAME_SEGOE_UI,
                    "Segoe UI",
                    config.font_name == FONT_NAME_SEGOE_UI,
                ),
                radio!(
                    CMD_FONT_NAME_ARIAL,
                    "Arial",
                    config.font_name == FONT_NAME_ARIAL
                ),
                radio!(
                    CMD_FONT_NAME_VERDANA,
                    "Verdana",
                    config.font_name == FONT_NAME_VERDANA,
                ),
                radio!(
                    CMD_FONT_NAME_TAHOMA,
                    "Tahoma",
                    config.font_name == FONT_NAME_TAHOMA
                ),
                radio!(
                    CMD_FONT_NAME_TIMES_NEW_ROMAN,
                    "Times New Roman",
                    config.font_name == FONT_NAME_TIMES_NEW_ROMAN,
                ),
                radio!(
                    CMD_FONT_NAME_CALIBRI,
                    "Calibri",
                    config.font_name == FONT_NAME_CALIBRI,
                ),
            ),
            submenu!(
                "Font Size",
                radio!(
                    CMD_FONT_SIZE_VERY_SMALL,
                    "Very Small",
                    config.font_size == FONT_SIZE_VERY_SMALL,
                ),
                radio!(
                    CMD_FONT_SIZE_SMALL,
                    "Small",
                    config.font_size == FONT_SIZE_SMALL
                ),
                radio!(
                    CMD_FONT_SIZE_MEDIUM,
                    "Medium",
                    config.font_size == FONT_SIZE_MEDIUM
                ),
                radio!(
                    CMD_FONT_SIZE_LARGE,
                    "Large",
                    config.font_size == FONT_SIZE_LARGE
                ),
                radio!(
                    CMD_FONT_SIZE_VERY_LARGE,
                    "Very Large",
                    config.font_size == FONT_SIZE_VERY_LARGE,
                ),
            ),
            submenu!(
                "Font Style",
                checkbox!(CMD_FONT_STYLE_BOLD, "Bold", config.bold),
                checkbox!(CMD_FONT_STYLE_ITALIC, "Italic", config.italic),
                checkbox!(CMD_FONT_STYLE_OUTLINE, "Outline", config.outline),
            ),
            separator!(),
            action!(CMD_QUIT, "Quit(&Q)"),
        ]
    }
}

macro_rules! cmd {
    ($category:expr, $item:expr, $cmd:ident) => {
        const $cmd: u32 = (0x100 * $category) + $item;
    };
}

cmd!(1, 1, CMD_CLEAR);
cmd!(2, 1, CMD_MODEL_SMALL_EN);
cmd!(2, 2, CMD_MODEL_MEDIUM_EN);
cmd!(2, 3, CMD_MODEL_LARGE_V3);
cmd!(3, 1, CMD_DELAY_LOWEST);
cmd!(3, 2, CMD_DELAY_LOW);
cmd!(3, 3, CMD_DELAY_MEDIUM);
cmd!(3, 4, CMD_DELAY_HIGH);
cmd!(3, 5, CMD_DELAY_HIGHEST);
cmd!(4, 1, CMD_TRANSPARENCY_0);
cmd!(4, 2, CMD_TRANSPARENCY_25);
cmd!(4, 3, CMD_TRANSPARENCY_50);
cmd!(4, 4, CMD_TRANSPARENCY_75);
cmd!(4, 5, CMD_TRANSPARENCY_100);
cmd!(5, 1, CMD_FONT_NAME_SEGOE_UI);
cmd!(5, 2, CMD_FONT_NAME_ARIAL);
cmd!(5, 3, CMD_FONT_NAME_VERDANA);
cmd!(5, 4, CMD_FONT_NAME_TAHOMA);
cmd!(5, 5, CMD_FONT_NAME_TIMES_NEW_ROMAN);
cmd!(5, 6, CMD_FONT_NAME_CALIBRI);
cmd!(6, 1, CMD_FONT_SIZE_VERY_SMALL);
cmd!(6, 2, CMD_FONT_SIZE_SMALL);
cmd!(6, 3, CMD_FONT_SIZE_MEDIUM);
cmd!(6, 4, CMD_FONT_SIZE_LARGE);
cmd!(6, 5, CMD_FONT_SIZE_VERY_LARGE);
cmd!(7, 1, CMD_FONT_STYLE_BOLD);
cmd!(7, 2, CMD_FONT_STYLE_ITALIC);
cmd!(7, 3, CMD_FONT_STYLE_OUTLINE);
cmd!(8, 1, CMD_QUIT);
