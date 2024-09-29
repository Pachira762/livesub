use anyhow::Result;
use windows::Win32::{
    Foundation::HWND,
    Graphics::{
        Direct2D::Common::D2D_RECT_F,
        DirectWrite::{
            IDWriteTextFormat, IDWriteTextLayout, DWRITE_LINE_METRICS, DWRITE_TEXT_METRICS,
        },
    },
};

use crate::gui::utils::CStr;

use super::context::Context;

pub struct Renderer {
    text: Vec<u16>,
    context: Context,
    format: Option<IDWriteTextFormat>,
    layout: Option<IDWriteTextLayout>,
    font_name: String,
    font_size: u32,
    font_style_bold: bool,
    font_style_italic: bool,
    font_style_outline: bool,
    opacity: f32,
    rect: D2D_RECT_F,
}

impl Renderer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hwnd: HWND,
        font_name: &str,
        font_size: u32,
        bold: bool,
        italic: bool,
        outline: bool,
        opacity: f32,
    ) -> Result<Self> {
        let (width, height) = (1024, 1024);
        let rect = D2D_RECT_F::new(0.0, 0.0, width as _, height as _).inner(8.0, 8.0);

        let context = Context::new(hwnd)?;
        let format = context.create_text_format(font_name, font_size, bold, italic)?;

        Ok(Self {
            text: vec![],
            context,
            format: Some(format),
            layout: None,
            font_name: font_name.into(),
            font_size,
            font_style_bold: bold,
            font_style_italic: italic,
            font_style_outline: outline,
            opacity,
            rect,
        })
    }

    pub fn draw(&mut self) -> Result<()> {
        if let Some(layout) = &self.layout {
            self.context.begin_draw(&[0.0, 0.0, 0.0, self.opacity]);
            self.context.enable_outline(self.font_style_outline);

            let viewport_height = self.rect.height();
            let layout_height = layout.metrics()?.height;
            let clip_and_offset = viewport_height < layout_height;

            if clip_and_offset {
                let mut clip_height = 0.0;
                for metrics in layout.line_metrics()?.iter().rev() {
                    if viewport_height < clip_height + metrics.baseline {
                        break;
                    }
                    clip_height += metrics.height;
                }

                let clip_rect = D2D_RECT_F {
                    left: self.rect.left - 1.0,
                    top: self.rect.bottom - clip_height + 1.0,
                    right: self.rect.right + 1.0,
                    bottom: self.rect.bottom + 1.0,
                };
                self.context.clip(&clip_rect);
            }

            let y = if clip_and_offset {
                self.rect.bottom - layout_height
            } else {
                self.rect.y()
            };
            self.context.draw_text(layout, self.rect.x(), y)?;

            if clip_and_offset {
                self.context.pop_clip();
            }

            self.context.end_draw()?;
        }
        Ok(())
    }

    pub fn set_text(&mut self, text: &str) {
        self.text = text.c_wstr();
        self.update_layout();
    }

    pub fn set_font_name(&mut self, font_name: &str) {
        self.font_name = font_name.into();
        self.update_format();
    }

    pub fn set_font_size(&mut self, font_size: u32) {
        self.font_size = font_size;
        self.update_format();
    }

    pub fn set_bold(&mut self, bold: bool) {
        self.font_style_bold = bold;
        self.update_format();
    }

    pub fn set_italic(&mut self, italic: bool) {
        self.font_style_italic = italic;
        self.update_format();
    }

    pub fn set_outline(&mut self, outline: bool) {
        self.font_style_outline = outline;
        _ = self.draw();
    }

    pub fn set_dpi(&mut self, dpi: u32) {
        self.context.set_dpi(dpi);
        _ = self.draw();
    }

    pub fn set_opacity(&mut self, opacity: f32) {
        self.opacity = opacity;
        _ = self.draw();
    }

    pub fn set_size(&mut self, width: u32, height: u32) -> Result<()> {
        self.context.set_size(width, height)?;

        let dpi = self.context.dpi();
        let width = 96.0 * width as f32 / dpi;
        let height = 96.0 * height as f32 / dpi;
        self.rect = D2D_RECT_F::new(0.0, 0.0, width, height).inner(8.0, 4.0);
        self.update_layout();
        Ok(())
    }

    fn update_format(&mut self) {
        self.layout = None;
        self.format = None;
        self.setup_text_format();
        self.setup_text_layout();
        _ = self.draw();
    }

    fn update_layout(&mut self) {
        self.layout = None;
        self.setup_text_layout();
        _ = self.draw();
    }

    fn setup_text_format(&mut self) {
        self.format = self
            .context
            .create_text_format(
                &self.font_name,
                self.font_size,
                self.font_style_bold,
                self.font_style_italic,
            )
            .ok();
    }

    fn setup_text_layout(&mut self) {
        self.layout = if let Some(format) = &self.format {
            self.context
                .create_text_layout(&self.text, format, self.rect.width(), self.rect.height())
                .ok()
        } else {
            None
        };
    }
}

trait RectF {
    fn new(x: f32, y: f32, width: f32, height: f32) -> Self;
    fn inner(&self, margin_x: f32, margin_y: f32) -> Self;
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn width(&self) -> f32;
    fn height(&self) -> f32;
}

impl RectF for D2D_RECT_F {
    fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            left: x,
            top: y,
            right: x + width,
            bottom: y + height,
        }
    }

    fn inner(&self, margin_x: f32, margin_y: f32) -> Self {
        Self {
            left: self.left + margin_x,
            top: self.top + margin_y,
            right: self.right - margin_x,
            bottom: self.bottom - margin_y,
        }
    }

    fn x(&self) -> f32 {
        self.left
    }

    fn y(&self) -> f32 {
        self.top
    }

    fn width(&self) -> f32 {
        self.right - self.left
    }

    fn height(&self) -> f32 {
        self.bottom - self.top
    }
}

trait TextLayout {
    fn metrics(&self) -> Result<DWRITE_TEXT_METRICS>;
    fn line_metrics(&self) -> Result<Vec<DWRITE_LINE_METRICS>>;
}

impl TextLayout for IDWriteTextLayout {
    fn metrics(&self) -> Result<DWRITE_TEXT_METRICS> {
        unsafe {
            let mut metrics = Default::default();
            self.GetMetrics(&mut metrics)?;
            Ok(metrics)
        }
    }

    fn line_metrics(&self) -> Result<Vec<DWRITE_LINE_METRICS>> {
        unsafe {
            let mut lines = 0;
            _ = self.GetLineMetrics(None, &mut lines);

            let mut metrics = vec![Default::default(); lines as usize];
            self.GetLineMetrics(Some(&mut metrics), &mut lines)?;

            Ok(metrics)
        }
    }
}
