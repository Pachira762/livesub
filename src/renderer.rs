use anyhow::Result;
use windows::{
    Foundation::Rect,
    UI::Composition::{CompositionStretch, Compositor, Desktop::DesktopWindowTarget},
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::{
            Direct2D::{
                Common::{D2D1_ALPHA_MODE_PREMULTIPLIED, D2D1_COLOR_F, D2D1_PIXEL_FORMAT},
                D2D1_BITMAP_OPTIONS_CANNOT_DRAW, D2D1_BITMAP_OPTIONS_TARGET,
                D2D1_BITMAP_PROPERTIES1, D2D1_DEVICE_CONTEXT_OPTIONS_NONE,
                D2D1_DRAW_TEXT_OPTIONS_NONE, D2D1_FACTORY_TYPE_SINGLE_THREADED, D2D1CreateFactory,
                ID2D1DeviceContext, ID2D1Factory2,
            },
            Direct3D::D3D_DRIVER_TYPE_HARDWARE,
            Direct3D11::{
                D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG,
                D3D11_CREATE_DEVICE_FLAG, D3D11_SDK_VERSION, D3D11CreateDevice,
            },
            DirectWrite::{
                DWRITE_FACTORY_TYPE_SHARED, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STYLE_ITALIC,
                DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_WEIGHT_NORMAL,
                DWriteCreateFactory, IDWriteFactory, IDWriteTextFormat, IDWriteTextLayout,
            },
            Dxgi::{
                Common::{
                    DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
                    DXGI_SAMPLE_DESC,
                },
                CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1,
                DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_EFFECT_FLIP_DISCARD,
                DXGI_USAGE_RENDER_TARGET_OUTPUT, IDXGIDevice, IDXGIFactory2, IDXGISurface2,
                IDXGISwapChain1,
            },
        },
        System::WinRT::Composition::{ICompositorDesktopInterop, ICompositorInterop},
        UI::{HiDpi::GetDpiForWindow, WindowsAndMessaging::GetClientRect},
    },
    core::{Interface, PCWSTR, w},
};
use windows_numerics::Vector2;

pub struct Renderer {
    hwnd: HWND,
    pub swap_chain: IDXGISwapChain1,
    pub context: ID2D1DeviceContext,
    pub dw_factory: IDWriteFactory,
    _compositor: Compositor,
    _window_target: DesktopWindowTarget,

    background: [f32; 4],
    font_name: String,
    font_size: f32,
    font_style_bold: bool,
    font_style_italic: bool,
    text_format: Option<IDWriteTextFormat>,

    text: String,
    layout: Rect,
    text_layout: Option<IDWriteTextLayout>,
}

impl Renderer {
    pub fn new(
        hwnd: HWND,
        font_name: &str,
        font_size: f32,
        font_style_bold: bool,
        font_style_italic: bool,
        background: [f32; 4],
    ) -> Result<Self> {
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

            let dpi = match GetDpiForWindow(hwnd) {
                0 => 96,
                dpi => dpi,
            };
            let (width, height) = {
                let mut rect = RECT::default();
                _ = GetClientRect(hwnd, &mut rect);
                (
                    (rect.right - rect.left).max(1) as u32,
                    (rect.bottom - rect.top).max(1) as u32,
                )
            };
            let swap_chain = {
                let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;

                let desc = DXGI_SWAP_CHAIN_DESC1 {
                    Width: width as _,
                    Height: height as _,
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

            let (_compositor, _window_target) = {
                let compositor = Compositor::new()?;

                let surface = {
                    let interop: ICompositorInterop = compositor.cast()?;
                    interop.CreateCompositionSurfaceForSwapChain(&swap_chain)?
                };

                let brush = compositor.CreateSurfaceBrushWithSurface(&surface)?;
                brush.SetStretch(CompositionStretch::Fill)?;

                let content = compositor.CreateSpriteVisual()?;
                content.SetRelativeSizeAdjustment(Vector2::one())?;
                content.SetBrush(&brush)?;

                let window_target = {
                    let interop: ICompositorDesktopInterop = compositor.cast()?;
                    interop.CreateDesktopWindowTarget(hwnd, true)?
                };
                window_target.SetRoot(&content)?;

                (compositor, window_target)
            };

            let d2d_factory: ID2D1Factory2 =
                D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;

            let context = {
                let device = d2d_factory.CreateDevice(&device)?;
                let context: ID2D1DeviceContext = device
                    .CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?
                    .cast()?;
                context
            };
            context.SetDpi(dpi as _, dpi as _);

            let dw_factory: IDWriteFactory = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;

            Ok(Self {
                hwnd,
                swap_chain,
                context,
                dw_factory,
                _compositor,
                _window_target,
                font_name: font_name.to_string(),
                font_size,
                font_style_bold,
                font_style_italic,
                background,
                text_format: None,
                text: String::new(),
                layout: Rect::default(),
                text_layout: None,
            })
        }
    }

    pub fn set_text(&mut self, text: String) {
        self.text = text.to_string();
        self.text_layout = None;
    }

    pub fn set_font_size(&mut self, font_size: f32) {
        self.font_size = font_size;
        self.text_format = None;
        self.text_layout = None;
    }

    pub fn set_font_style_bold(&mut self, bold: bool) {
        self.font_style_bold = bold;
        self.text_format = None;
        self.text_layout = None;
    }

    pub fn set_font_style_italic(&mut self, italic: bool) {
        self.font_style_italic = italic;
        self.text_format = None;
        self.text_layout = None;
    }

    pub fn set_background(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.background[0] = r;
        self.background[1] = g;
        self.background[2] = b;
        self.background[3] = a;
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            if self.text_format.is_none() {
                let font_family: Vec<_> = self.font_name.encode_utf16().chain([0]).collect();
                let font_weight = if self.font_style_bold {
                    DWRITE_FONT_WEIGHT_BOLD
                } else {
                    DWRITE_FONT_WEIGHT_NORMAL
                };
                let font_style = if self.font_style_italic {
                    DWRITE_FONT_STYLE_ITALIC
                } else {
                    DWRITE_FONT_STYLE_NORMAL
                };
                self.text_format = Some(self.dw_factory.CreateTextFormat(
                    PCWSTR(font_family.as_ptr()),
                    None,
                    font_weight,
                    font_style,
                    DWRITE_FONT_STRETCH_NORMAL,
                    self.font_size,
                    w!(""),
                )?);
            }

            if self.text_layout.is_none() {
                let text: Vec<_> = self.text.encode_utf16().collect();
                self.text_layout = Some(self.dw_factory.CreateTextLayout(
                    &text,
                    self.text_format.as_ref().unwrap(),
                    self.layout.Width,
                    self.layout.Height,
                )?);
            }

            self.context.BeginDraw();

            let background = D2D1_COLOR_F {
                r: self.background[0],
                g: self.background[1],
                b: self.background[2],
                a: self.background[3],
            };
            self.context.Clear(Some(&background));

            if let Some(text_layout) = &self.text_layout {
                let white = 0.95;
                let brush = self.context.CreateSolidColorBrush(
                    &D2D1_COLOR_F {
                        r: white,
                        g: white,
                        b: white,
                        a: 1.0,
                    },
                    None,
                )?;

                let metrics = {
                    let mut metrics = Default::default();
                    text_layout.GetMetrics(&mut metrics)?;
                    metrics
                };
                let x = self.layout.X;
                let y = if metrics.height < self.layout.Height {
                    self.layout.Y
                } else {
                    (self.layout.Y + self.layout.Height) - metrics.height
                };

                self.context.DrawTextLayout(
                    Vector2 { X: x, Y: y },
                    text_layout,
                    &brush,
                    D2D1_DRAW_TEXT_OPTIONS_NONE,
                );
            }

            self.context.EndDraw(None, None)?;
            self.swap_chain.Present(1, DXGI_PRESENT(0)).ok()?;

            Ok(())
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            if width == 0 || height == 0 {
                return Ok(());
            }
            let dpi = GetDpiForWindow(self.hwnd) as f32;

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
                self.context.CreateBitmapFromDxgiSurface(
                    &surface,
                    Some(&D2D1_BITMAP_PROPERTIES1 {
                        pixelFormat: D2D1_PIXEL_FORMAT {
                            format: DXGI_FORMAT_B8G8R8A8_UNORM,
                            alphaMode: D2D1_ALPHA_MODE_PREMULTIPLIED,
                        },
                        dpiX: dpi,
                        dpiY: dpi,
                        bitmapOptions: D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
                        ..Default::default()
                    }),
                )?
            };
            self.context.SetTarget(&target);

            const MARGIN_X: f32 = 4.0;
            const MARGIN_Y: f32 = 4.0;
            self.layout.X = MARGIN_X;
            self.layout.Y = MARGIN_Y;
            self.layout.Width = width as f32 * 96.0 / dpi - 2.0 * MARGIN_X;
            self.layout.Height = height as f32 * 96.0 / dpi - 2.0 * MARGIN_Y;

            self.text_layout = None;

            Ok(())
        }
    }
}
