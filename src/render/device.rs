use anyhow::{Error as E, Result};
use windows::core::Interface;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Direct2D::Common::D2D1_COLOR_F;
use windows::Win32::Graphics::Direct2D::{
    D2D1CreateFactory, ID2D1Bitmap1, ID2D1Device1, ID2D1DeviceContext, ID2D1Factory2,
    D2D1_DEVICE_CONTEXT_OPTIONS_NONE, D2D1_FACTORY_TYPE_SINGLE_THREADED,
};
use windows::Win32::Graphics::DirectComposition::{
    DCompositionCreateDevice, IDCompositionDevice, IDCompositionTarget,
};
use windows::Win32::Graphics::DirectWrite::{
    DWriteCreateFactory, IDWriteFactory, DWRITE_FACTORY_TYPE_SHARED,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_PREMULTIPLIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN,
    DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIDevice, IDXGIFactory2, IDXGISurface2, IDXGISwapChain1,
    DXGI_CREATE_FACTORY_FLAGS, DXGI_PRESENT, DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
};
use windows::Win32::Graphics::{
    Direct3D::D3D_DRIVER_TYPE_HARDWARE,
    Direct3D11::{D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION},
};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

pub struct Device {
    #[allow(unused)]
    pub device_dxgi: IDXGIDevice,

    #[allow(unused)]
    pub swap_chain: IDXGISwapChain1,

    #[allow(unused)]
    pub device_dcomp: IDCompositionDevice,

    #[allow(unused)]
    pub target: IDCompositionTarget,

    #[allow(unused)]
    pub factory_2d: ID2D1Factory2,

    #[allow(unused)]
    pub device_2d: ID2D1Device1,

    #[allow(unused)]
    pub dc: ID2D1DeviceContext,

    #[allow(unused)]
    pub factory_dw: IDWriteFactory,
}

impl Device {
    pub fn new(hwnd: HWND) -> Result<Self> {
        unsafe {
            let mut device_3d = None;
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                None,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device_3d),
                None,
                None,
            )?;
            let device_3d = device_3d.unwrap();
            let device_dxgi: IDXGIDevice = device_3d.cast()?;
            let factory_dxgi: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;

            let (width, height) = {
                let mut rc = RECT::default();
                _ = GetClientRect(hwnd, &mut rc);
                ((rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32)
            };

            let swap_chain = factory_dxgi.CreateSwapChainForComposition(
                &device_dxgi,
                &DXGI_SWAP_CHAIN_DESC1 {
                    Width: width,
                    Height: height,
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
                },
                None,
            )?;

            let device_dcomp: IDCompositionDevice = DCompositionCreateDevice(&device_dxgi)?;
            let target = device_dcomp.CreateTargetForHwnd(hwnd, true)?;
            let visual = device_dcomp.CreateVisual()?;

            visual.SetContent(&swap_chain)?;
            target.SetRoot(&visual)?;
            device_dcomp.Commit()?;

            let factory_2d: ID2D1Factory2 =
                D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, None)?;
            let device_2d = factory_2d.CreateDevice(&device_dxgi)?;
            let dc: ID2D1DeviceContext = device_2d
                .CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE)?
                .cast()?;

            dc.SetTarget(&create_target(&swap_chain, &dc)?);

            let factory_dw = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED)?;

            Ok(Self {
                device_dxgi,
                swap_chain,
                device_dcomp,
                target,
                factory_2d,
                device_2d,
                dc,
                factory_dw,
            })
        }
    }

    pub fn begin_draw(&self, clear_color: &[f32]) -> &ID2D1DeviceContext {
        unsafe {
            self.dc.BeginDraw();

            self.dc.Clear(Some(&D2D1_COLOR_F {
                r: clear_color[0],
                g: clear_color[1],
                b: clear_color[2],
                a: clear_color[3],
            }));

            &self.dc
        }
    }

    pub fn end_draw(&self) -> Result<()> {
        unsafe {
            self.dc.EndDraw(None, None)?;
            self.swap_chain
                .Present(1, DXGI_PRESENT(0))
                .ok()
                .map_err(E::msg)
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        unsafe {
            self.dc.SetTarget(None);

            self.swap_chain.ResizeBuffers(
                0,
                width,
                height,
                DXGI_FORMAT_UNKNOWN,
                DXGI_SWAP_CHAIN_FLAG(0),
            )?;

            self.dc
                .SetTarget(&create_target(&self.swap_chain, &self.dc)?);
        }
        Ok(())
    }
}

fn create_target(swap_chain: &IDXGISwapChain1, dc: &ID2D1DeviceContext) -> Result<ID2D1Bitmap1> {
    unsafe {
        let surface: IDXGISurface2 = swap_chain.GetBuffer(0)?;
        dc.CreateBitmapFromDxgiSurface(&surface, None)
            .map_err(E::msg)
    }
}
