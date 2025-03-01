use anyhow::{Error as E, Result};
use windows::{
    Win32::{
        Foundation::{FALSE, HWND},
        Graphics::{
            Direct2D::{
                Common::{D2D1_COLOR_F, D2D_RECT_F},
                D2D1CreateFactory, ID2D1DeviceContext, ID2D1Factory2, ID2D1SolidColorBrush,
                D2D1_ANTIALIAS_MODE_PER_PRIMITIVE, D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
                D2D1_FACTORY_TYPE_SINGLE_THREADED,
            },
            Direct3D::D3D_DRIVER_TYPE_HARDWARE,
            Direct3D11::{
                D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG,
                D3D11_CREATE_DEVICE_FLAG, D3D11_SDK_VERSION,
            },
            DirectWrite::{
                DWriteCreateFactory, IDWriteFactory, IDWritePixelSnapping_Impl, IDWriteTextFormat,
                IDWriteTextLayout, IDWriteTextRenderer, IDWriteTextRenderer_Impl,
                DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_NORMAL,
                DWRITE_FONT_STYLE_OBLIQUE, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_WEIGHT_REGULAR,
                DWRITE_GLYPH_RUN, DWRITE_GLYPH_RUN_DESCRIPTION, DWRITE_MATRIX,
                DWRITE_MEASURING_MODE, DWRITE_STRIKETHROUGH, DWRITE_UNDERLINE,
            },
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
                    DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory2, IDXGIDevice, IDXGIFactory2, IDXGISurface2, IDXGISwapChain1,
                DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1,
                DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                DXGI_USAGE_RENDER_TARGET_OUTPUT,
            },
        },
        System::WinRT::Composition::{ICompositorDesktopInterop, ICompositorInterop},
    },
    UI::Composition::{CompositionStretch, Compositor, Desktop::DesktopWindowTarget},
};
use windows_core::{implement, w, Interface as _, BOOL, PCWSTR};
use windows_numerics::{Matrix3x2, Vector2};

use crate::gui::utils::{CStr, Hwnd};

pub struct Context {
    pub swap_chain: IDXGISwapChain1,
    pub context: ID2D1DeviceContext,
    pub dw_factory: IDWriteFactory,
    pub renderer: TextRenderer,
    _compositor: Compositor,
    _window_targets: Vec<DesktopWindowTarget>,
}

impl Context {
    pub fn new(hwnd: HWND) -> Result<Self> {
        unsafe {
            let device: IDXGIDevice = {
                let device_3d = {
                    let flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT
                        | if cfg!(debug_assertions) {
                            D3D11_CREATE_DEVICE_DEBUG
                        } else {
                            D3D11_CREATE_DEVICE_FLAG(0)
                        };
                    let mut device_3d = None;
                    D3D11CreateDevice(
                        None,
                        D3D_DRIVER_TYPE_HARDWARE,
                        Default::default(),
                        flags,
                        None,
                        D3D11_SDK_VERSION,
                        Some(&mut device_3d),
                        None,
                        None,
                    )?;
                    device_3d.unwrap()
                };
                device_3d.cast()?
            };

            let swap_chain = {
                let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;

                let desc = DXGI_SWAP_CHAIN_DESC1 {
                    Width: 1024,
                    Height: 1024,
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: 2,
                    SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                    AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
                    ..Default::default()
                };
                factory.CreateSwapChainForComposition(&device, &desc, None)?
            };

            let (_compositor, _window_targets) = {
                let compositor = Compositor::new()?;
                let window_target = {
                    let interop: ICompositorDesktopInterop = compositor.cast()?;
                    interop.CreateDesktopWindowTarget(hwnd, true)?
                };
                let content = compositor.CreateSpriteVisual()?;
                let surface = {
                    let interop: ICompositorInterop = compositor.cast()?;
                    interop.CreateCompositionSurfaceForSwapChain(&swap_chain)?
                };
                let brush = compositor.CreateSurfaceBrushWithSurface(&surface)?;
                brush.SetStretch(CompositionStretch::UniformToFill)?;

                content.SetRelativeSizeAdjustment(Vector2::one())?;
                content.SetBrush(&brush)?;
                window_target.SetRoot(&content)?;

                (compositor, vec![window_target])
            };

            let d2d_factory: ID2D1Factory2 =
                D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;

            let context = {
                let device = d2d_factory.CreateDevice(&device)?;
                let context: ID2D1DeviceContext = device
                    .CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?
                    .cast()?;

                let dpi = hwnd.dpi();
                context.SetDpi(dpi as _, dpi as _);

                context
            };

            let target = {
                let surface: IDXGISurface2 = swap_chain.GetBuffer(0)?;
                context.CreateBitmapFromDxgiSurface(&surface, None)?
            };
            context.SetTarget(&target);

            let dw_factory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;
            let renderer = TextRenderer::new(d2d_factory, context.clone())?;

            Ok(Self {
                swap_chain,
                context,
                dw_factory,
                renderer,
                _compositor,
                _window_targets,
            })
        }
    }

    pub fn begin_draw(&self, clear_color: &[f32]) {
        unsafe {
            self.context.BeginDraw();

            self.context.Clear(Some(&D2D1_COLOR_F {
                r: clear_color[0],
                g: clear_color[1],
                b: clear_color[2],
                a: clear_color[3],
            }));
        }
    }

    pub fn end_draw(&self) -> Result<()> {
        unsafe {
            self.context.EndDraw(None, None)?;

            self.swap_chain
                .Present(1, DXGI_PRESENT(0))
                .ok()
                .map_err(E::msg)
        }
    }

    pub fn draw_text(&self, layout: &IDWriteTextLayout, x: f32, y: f32) -> Result<()> {
        unsafe {
            let context = Some(self.context.as_raw() as *const _);
            let renderer: IDWriteTextRenderer = self.renderer.clone().into();
            layout
                .Draw(context, &renderer, x, y)
                .map_err(anyhow::Error::msg)
        }
    }

    pub fn clip(&self, rect: &D2D_RECT_F) {
        unsafe {
            self.context
                .PushAxisAlignedClip(rect, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);
        }
    }

    pub fn pop_clip(&self) {
        unsafe {
            self.context.PopAxisAlignedClip();
        }
    }

    pub fn enable_outline(&mut self, outline: bool) {
        self.renderer.enable_outline(outline);
    }

    pub fn set_size(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.context.SetTarget(None);

            self.swap_chain.ResizeBuffers(
                0,
                width,
                height,
                DXGI_FORMAT_UNKNOWN,
                DXGI_SWAP_CHAIN_FLAG(0),
            )?;

            let target = {
                let surface: IDXGISurface2 = self.swap_chain.GetBuffer(0)?;
                self.context.CreateBitmapFromDxgiSurface(&surface, None)?
            };
            self.context.SetTarget(&target);
        }
        Ok(())
    }

    pub fn dpi(&self) -> f32 {
        unsafe {
            let mut x = 0.0;
            let mut y = 0.0;
            self.context.GetDpi(&mut x, &mut y);
            y
        }
    }

    pub fn set_dpi(&mut self, dpi: u32) {
        unsafe {
            self.context.SetDpi(dpi as _, dpi as _);
        }
    }

    pub fn create_text_format(
        &self,
        font_name: &str,
        font_size: u32,
        bold: bool,
        italic: bool,
    ) -> Result<IDWriteTextFormat> {
        unsafe {
            let family = font_name.c_wstr();

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

            self.dw_factory
                .CreateTextFormat(
                    PCWSTR(family.as_ptr()),
                    None,
                    weight,
                    style,
                    DWRITE_FONT_STRETCH_NORMAL,
                    font_size as _,
                    w!(""),
                )
                .map_err(anyhow::Error::msg)
        }
    }

    pub fn create_text_layout(
        &self,
        text: &[u16],
        format: &IDWriteTextFormat,
        width: f32,
        height: f32,
    ) -> Result<IDWriteTextLayout> {
        unsafe {
            self.dw_factory
                .CreateTextLayout(text, format, width, height)
                .map_err(anyhow::Error::msg)
        }
    }
}

#[derive(Clone)]
#[implement(IDWriteTextRenderer)]
pub struct TextRenderer {
    factory: ID2D1Factory2,
    dc: ID2D1DeviceContext,
    outline_brush: ID2D1SolidColorBrush,
    fill_brush: ID2D1SolidColorBrush,
    outline: bool,
}

impl TextRenderer {
    pub fn new(factory: ID2D1Factory2, dc: ID2D1DeviceContext) -> Result<Self> {
        const BLACK_LEVEL: f32 = 0.01;
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

impl IDWriteTextRenderer_Impl for TextRenderer_Impl {
    fn DrawGlyphRun(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        baselineoriginx: f32,
        baselineoriginy: f32,
        _measuringmode: DWRITE_MEASURING_MODE,
        glyphrun: *const DWRITE_GLYPH_RUN,
        _glyphrundescription: *const DWRITE_GLYPH_RUN_DESCRIPTION,
        _clientdrawingeffect: windows_core::Ref<'_, windows_core::IUnknown>,
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
                glyphrun.isSideways.into(),
                if (glyphrun.bidiLevel as i32 % 2) == 0 {
                    false
                } else {
                    true
                },
                &sink,
            )?;
            sink.Close()?;

            let matrix = Matrix3x2::translation(baselineoriginx, baselineoriginy);
            let geometory = self.factory.CreateTransformedGeometry(&geometry, &matrix)?;

            if self.outline {
                self.dc
                    .DrawGeometry(&geometory, &self.outline_brush, 4.0, None);
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
        _clientdrawingeffect: windows_core::Ref<'_, windows_core::IUnknown>,
    ) -> ::windows::core::Result<()> {
        todo!()
    }

    fn DrawStrikethrough(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        _baselineoriginx: f32,
        _baselineoriginy: f32,
        _strikethrough: *const DWRITE_STRIKETHROUGH,
        _clientdrawingeffect: windows_core::Ref<'_, windows_core::IUnknown>,
    ) -> ::windows::core::Result<()> {
        todo!()
    }

    fn DrawInlineObject(
        &self,
        _clientdrawingcontext: *const ::core::ffi::c_void,
        _originx: f32,
        _originy: f32,
        _inlineobject: windows_core::Ref<
            '_,
            windows::Win32::Graphics::DirectWrite::IDWriteInlineObject,
        >,
        _issideways: BOOL,
        _isrighttoleft: BOOL,
        _clientdrawingeffect: windows_core::Ref<'_, windows_core::IUnknown>,
    ) -> ::windows::core::Result<()> {
        todo!()
    }
}

impl IDWritePixelSnapping_Impl for TextRenderer_Impl {
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
            Ok(y / 96.0)
        }
    }
}
