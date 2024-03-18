use anyhow::Result;
use windows::{
    core::*,
    Foundation::Numerics::Matrix3x2,
    Win32::{
        Foundation::*,
        Graphics::{
            Direct2D::{Common::*, *},
            DirectWrite::*,
        },
    },
};

#[derive(Clone)]
#[implement(IDWriteTextRenderer)]
pub struct TextRenderer {
    factory: ID2D1Factory2,
    dc: ID2D1DeviceContext,
    outline_brush: ID2D1SolidColorBrush,
    fill_brush: ID2D1SolidColorBrush,
    pub outline: bool,
}

impl TextRenderer {
    pub fn new(factory: ID2D1Factory2, dc: ID2D1DeviceContext) -> Result<Self> {
        const BLACK_LEVEL: f32 = 0.0;
        const WHITE_LEVEL: f32 = 1.0;

        unsafe {
            let outline_brush = dc.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: BLACK_LEVEL,
                    g: BLACK_LEVEL,
                    b: BLACK_LEVEL,
                    a: 1.0,
                },
                None,
            )?;

            let fill_brush = dc.CreateSolidColorBrush(
                &D2D1_COLOR_F {
                    r: WHITE_LEVEL,
                    g: WHITE_LEVEL,
                    b: WHITE_LEVEL,
                    a: 1.0,
                },
                None,
            )?;

            Ok(Self {
                factory,
                dc,
                outline_brush,
                fill_brush,
                outline: false,
            })
        }
    }

    pub fn enable_outline(&mut self, enable: bool) {
        self.outline = enable;
    }
}

impl IDWriteTextRenderer_Impl for TextRenderer {
    fn DrawGlyphRun(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        baselineoriginx: f32,
        baselineoriginy: f32,
        _measuringmode: DWRITE_MEASURING_MODE,
        glyphrun: *const DWRITE_GLYPH_RUN,
        _glyphrundescription: *const DWRITE_GLYPH_RUN_DESCRIPTION,
        _clientdrawingeffect: ::core::option::Option<&IUnknown>,
    ) -> ::windows::core::Result<()> {
        unsafe {
            let geometry = self.factory.CreatePathGeometry()?;
            let sink = geometry.Open()?;
            let glyphrun = &*glyphrun;
            let font_face = glyphrun.fontFace.as_ref().unwrap();

            font_face.GetGlyphRunOutline(
                glyphrun.fontEmSize,
                glyphrun.glyphIndices,
                Some(glyphrun.glyphAdvances),
                Some(glyphrun.glyphOffsets),
                glyphrun.glyphCount,
                glyphrun.isSideways,
                BOOL(glyphrun.bidiLevel as i32 % 2),
                &sink,
            )?;
            sink.Close()?;

            let matrix = Matrix3x2::translation(baselineoriginx, baselineoriginy);
            let geometory = self.factory.CreateTransformedGeometry(&geometry, &matrix)?;

            if self.outline {
                self.dc
                    .DrawGeometry(&geometory, &self.outline_brush, 3.0, None);
            }

            self.dc.FillGeometry(&geometory, &self.fill_brush, None);
        }
        Ok(())
    }

    fn DrawUnderline(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _underline: *const DWRITE_UNDERLINE,
        _clientdrawingeffect: ::core::option::Option<&::windows::core::IUnknown>,
    ) -> ::windows::core::Result<()> {
        todo!()
    }

    fn DrawStrikethrough(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _strikethrough: *const DWRITE_STRIKETHROUGH,
        _clientdrawingeffect: ::core::option::Option<&::windows::core::IUnknown>,
    ) -> ::windows::core::Result<()> {
        todo!()
    }

    fn DrawInlineObject(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        _originx: f32,
        _originy: f32,
        _inlineobject: ::core::option::Option<&IDWriteInlineObject>,
        _issideways: BOOL,
        _isrighttoleft: BOOL,
        _clientdrawingeffect: ::core::option::Option<&::windows::core::IUnknown>,
    ) -> ::windows::core::Result<()> {
        todo!()
    }
}

impl IDWritePixelSnapping_Impl for TextRenderer {
    fn IsPixelSnappingDisabled(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
    ) -> ::windows::core::Result<BOOL> {
        Ok(FALSE)
    }

    fn GetCurrentTransform(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        transform: *mut DWRITE_MATRIX,
    ) -> ::windows::core::Result<()> {
        unsafe {
            self.dc.GetTransform(transform as *mut _);
        }
        Ok(())
    }

    fn GetPixelsPerDip(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
    ) -> ::windows::core::Result<f32> {
        unsafe {
            let mut x = 0.0;
            let mut y = 0.0;
            self.dc.GetDpi(&mut x, &mut y);
            Ok(x / 96.0)
        }
    }
}
