use anyhow::{Error as E, Result};
use windows::{
    core::{w, PCWSTR},
    Win32::Graphics::DirectWrite::{
        IDWriteFactory, IDWriteTextFormat, IDWriteTextLayout, DWRITE_FONT_STRETCH_NORMAL,
        DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STYLE_OBLIQUE, DWRITE_FONT_WEIGHT_BOLD,
        DWRITE_FONT_WEIGHT_REGULAR,
    },
};

pub struct TextStyle {
    factory: IDWriteFactory,
    font_name: String,
    font_size: f32,
    bold: bool,
    italic: bool,
    dpi: u32,
    text_format: IDWriteTextFormat,
}

impl TextStyle {
    pub fn new(factory: IDWriteFactory, font_name: &str, font_size: f32, dpi: u32) -> Result<Self> {
        let bold = false;
        let italic = false;
        let text_format = create_text_format(&factory, font_name, font_size, bold, italic, dpi)?;

        Ok(Self {
            factory,
            font_name: font_name.to_owned(),
            font_size,
            bold,
            italic,
            dpi,
            text_format,
        })
    }

    pub fn set_font_name(&mut self, font_name: &str) -> Result<()> {
        self.font_name = font_name.to_owned();
        self.update_text_format()
    }

    pub fn set_font_size(&mut self, font_size: f32) -> Result<()> {
        self.font_size = font_size;
        self.update_text_format()
    }

    pub fn set_bold(&mut self, bold: bool) -> Result<()> {
        self.bold = bold;
        self.update_text_format()
    }

    pub fn set_italic(&mut self, italic: bool) -> Result<()> {
        self.italic = italic;
        self.update_text_format()
    }

    pub fn set_dpi(&mut self, dpi: u32) -> Result<()> {
        self.dpi = dpi;
        self.update_text_format()
    }

    pub fn create_text_layout(
        &self,
        text: &[u16],
        width: f32,
        height: f32,
    ) -> Result<IDWriteTextLayout> {
        unsafe {
            self.factory
                .CreateTextLayout(text, &self.text_format, width, height)
                .map_err(E::msg)
        }
    }

    fn update_text_format(&mut self) -> Result<()> {
        self.text_format = create_text_format(
            &self.factory,
            &self.font_name,
            self.font_size,
            self.bold,
            self.italic,
            self.dpi,
        )?;
        Ok(())
    }
}

fn create_text_format(
    factory: &IDWriteFactory,
    font_name: &str,
    font_size: f32,
    bold: bool,
    italic: bool,
    dpi: u32,
) -> Result<IDWriteTextFormat> {
    unsafe {
        let font_name = font_name.to_owned() + "\0";
        let font_name: Vec<_> = font_name.encode_utf16().collect();

        let weight = if bold {
            DWRITE_FONT_WEIGHT_BOLD
        } else {
            DWRITE_FONT_WEIGHT_REGULAR
        };

        let style = if italic {
            DWRITE_FONT_STYLE_OBLIQUE
        } else {
            DWRITE_FONT_STYLE_NORMAL
        };

        factory
            .CreateTextFormat(
                PCWSTR(font_name.as_ptr()),
                None,
                weight,
                style,
                DWRITE_FONT_STRETCH_NORMAL,
                font_size * dpi as f32 / 96.0,
                w!(""),
            )
            .map_err(E::msg)
    }
}
