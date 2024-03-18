use anyhow::Result;
use windows::Win32::{
    Foundation::*,
    Globalization::{MultiByteToWideChar, CP_UTF8, MULTI_BYTE_TO_WIDE_CHAR_FLAGS},
    Graphics::{Direct2D::Common::*, DirectWrite::*},
    UI::{HiDpi::GetDpiForWindow, WindowsAndMessaging::GetWindowRect},
};

use super::{device::Device, style::TextStyle, text::TextRenderer};

const MARGIN_X: f32 = 12.0;
const MARGIN_Y: f32 = 12.0;

pub struct Renderer {
    device: Device,
    text: Vec<u16>,
    text_style: TextStyle,
    text_renderer: TextRenderer,
    text_layout: Option<IDWriteTextLayout>,
    rect: D2D_RECT_F,
    transparency: f32,
    dpi: u32,
}

impl Renderer {
    pub fn new(hwnd: HWND, font_name: &str, font_size: f32, transparency: f32) -> Result<Self> {
        unsafe {
            let dpi = GetDpiForWindow(hwnd);
            let device = Device::new(hwnd)?;

            let text_style = TextStyle::new(device.factory_dw.clone(), font_name, font_size, dpi)?;
            let text_renderer = TextRenderer::new(device.factory_2d.clone(), device.dc.clone())?;
            let text_layout = None;

            let (width, height) = {
                let mut rc = RECT::default();
                _ = GetWindowRect(hwnd, &mut rc);
                (rc.right - rc.left, rc.bottom - rc.top)
            };

            let scale = dpi as f32 / 96.0;
            let rect = D2D_RECT_F {
                left: scale * MARGIN_X,
                top: scale * MARGIN_Y,
                right: width as f32 - scale * MARGIN_X,
                bottom: height as f32 - scale * MARGIN_Y,
            };

            Ok(Self {
                device,
                text: vec![],
                text_style,
                text_layout,
                text_renderer,
                rect,
                transparency,
                dpi,
            })
        }
    }

    pub fn draw(&mut self) {
        unsafe {
            let dc = self.device.begin_draw(&[0.0, 0.0, 0.0, self.transparency]);

            if let Some(text_layout) = &self.text_layout {
                let mut metrics = Default::default();
                _ = text_layout.GetMetrics(&mut metrics);

                let offset = if metrics.height < metrics.layoutHeight {
                    0.0
                } else {
                    metrics.height - metrics.layoutHeight
                };

                let renderer: IDWriteTextRenderer = self.text_renderer.clone().into();
                let _ = text_layout.Draw(
                    Some(dc as *const _ as _),
                    &renderer,
                    self.rect.left,
                    self.rect.top - offset,
                );
            }

            self.device.end_draw().expect("failed to end draw.");
        }
    }

    pub fn set_text(&mut self, text: &str) {
        unsafe {
            let n = MultiByteToWideChar(
                CP_UTF8,
                MULTI_BYTE_TO_WIDE_CHAR_FLAGS(0),
                text.as_bytes(),
                None,
            );

            self.text.resize(n as usize, 0);

            MultiByteToWideChar(
                CP_UTF8,
                MULTI_BYTE_TO_WIDE_CHAR_FLAGS(0),
                text.as_bytes(),
                Some(&mut self.text),
            );
        }

        self.update_text_layout();
    }

    pub fn set_font_name(&mut self, font_name: &str) {
        _ = self.text_style.set_font_name(font_name);
        self.update_text_layout();
    }

    pub fn set_font_size(&mut self, font_size: f32) {
        _ = self.text_style.set_font_size(font_size);
        self.update_text_layout();
    }

    pub fn set_bold(&mut self, bold: bool) {
        _ = self.text_style.set_bold(bold);
        self.update_text_layout();
    }

    pub fn set_italic(&mut self, italic: bool) {
        _ = self.text_style.set_italic(italic);
        self.update_text_layout();
    }

    pub fn set_outline(&mut self, outline: bool) {
        self.text_renderer.enable_outline(outline);
    }

    pub fn set_dpi(&mut self, dpi: u32) {
        self.dpi = dpi;
        _ = self.text_style.set_dpi(dpi);
        self.update_text_layout();
    }

    pub fn set_transparency(&mut self, transparency: f32) {
        self.transparency = transparency;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.device
            .resize(width, height)
            .expect("failed to resize render device.");

        let scale = self.dpi as f32 / 96.0;
        self.rect.right = width as f32 - scale * MARGIN_X;
        self.rect.bottom = height as f32 - scale * MARGIN_Y;

        self.update_text_layout();
    }

    fn update_text_layout(&mut self) {
        let width = self.rect.right - self.rect.left;
        let height = self.rect.bottom - self.rect.top;

        self.text_layout = self
            .text_style
            .create_text_layout(&self.text, width, height)
            .ok();
    }
}
