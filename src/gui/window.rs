#![allow(unused, clippy::too_many_arguments)]

use std::ffi::c_void;

use anyhow::Result;
use windows::{
    core::{Param, PCSTR, PCWSTR},
    Win32::{
        Foundation::*,
        Graphics::{
            Dwm::{
                DwmDefWindowProc, DwmEnableBlurBehindWindow, DwmExtendFrameIntoClientArea,
                DwmGetWindowAttribute, DwmSetWindowAttribute, DWMWA_CAPTION_BUTTON_BOUNDS,
                DWMWINDOWATTRIBUTE, DWM_BB_ENABLE, DWM_BLURBEHIND,
            },
            Gdi::{UpdateWindow, HBRUSH},
        },
        UI::{
            Controls::{
                SetScrollInfo, SetWindowTheme, BST_CHECKED, BST_UNCHECKED, MARGINS, TBM_SETPOS,
                TBM_SETRANGEMAX, TBM_SETRANGEMIN,
            },
            WindowsAndMessaging::*,
        },
    },
};
use windows_core::s;

use super::utils::{self, Hinstance, Hwnd as _, Rect};

pub type WndProc = unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;

pub trait WindowClass: Sized {
    fn new() -> Self;
    fn set_style(&mut self, style: WNDCLASS_STYLES) -> &mut Self;
    fn set_wndproc(&mut self, wndproc: WndProc) -> &mut Self;
    fn set_icon(&mut self, icon: HICON) -> &mut Self;
    fn set_cursor(&mut self, cursor: HCURSOR) -> &mut Self;
    fn set_brush(&mut self, brush: HBRUSH) -> &mut Self;
    fn set_name(&mut self, classname: PCSTR) -> &mut Self;
    fn register(&mut self) -> Result<()>;
}

impl WindowClass for WNDCLASSEXA {
    fn new() -> WNDCLASSEXA {
        WNDCLASSEXA {
            cbSize: std::mem::size_of::<WNDCLASSEXA>() as u32,
            hInstance: HINSTANCE::get(),
            ..Default::default()
        }
    }

    fn set_style(&mut self, style: WNDCLASS_STYLES) -> &mut Self {
        self.style = style;
        self
    }

    fn set_wndproc(&mut self, wndproc: WndProc) -> &mut Self {
        self.lpfnWndProc = Some(wndproc);
        self
    }

    fn set_icon(&mut self, icon: HICON) -> &mut Self {
        self.hIcon = icon;
        self
    }

    fn set_cursor(&mut self, cursor: HCURSOR) -> &mut Self {
        self.hCursor = cursor;
        self
    }

    fn set_brush(&mut self, brush: HBRUSH) -> &mut Self {
        self.hbrBackground = brush;
        self
    }

    fn set_name(&mut self, classname: PCSTR) -> &mut Self {
        self.lpszClassName = classname;
        self
    }

    fn register(&mut self) -> Result<()> {
        if unsafe { RegisterClassExA(self) } == 0 {
            anyhow::bail!(windows::core::Error::from_win32())
        } else {
            Ok(())
        }
    }
}

pub trait Window {
    fn new(hwnd: HWND, cs: &mut CREATESTRUCTA) -> Result<Box<Self>>;
    fn wndproc(&mut self, hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> Option<LRESULT>;
}

pub unsafe extern "system" fn wndproc<T: Window>(
    hwnd: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
) -> LRESULT {
    if let Some(result) = hwnd.dwm_def_proc(msg, wp, lp) {
        return result;
    }

    if let Some(mut window) = std::ptr::NonNull::new(hwnd.user_data() as *mut T) {
        if let Some(result) = window.as_mut().wndproc(hwnd, msg, wp, lp) {
            return result;
        }
    }

    match msg {
        WM_NCCREATE => {
            let cs = unsafe { (lp.0 as *mut CREATESTRUCTA).as_mut().unwrap() };
            match T::new(hwnd, cs) {
                Ok(window) => {
                    let window = Box::leak(window);
                    hwnd.set_user_data(window as *mut _ as _);

                    LRESULT(1)
                }
                Err(e) => {
                    unsafe {
                        let text = format!("{e:?}\0");
                        MessageBoxA(None, PCSTR(text.as_ptr()), s!("error"), MB_OK);
                    }
                    LRESULT(0)
                }
            }
        }
        WM_CLOSE => {
            hwnd.destroy();
            LRESULT(0)
        }
        WM_DESTROY => {
            utils::quit(0);
            LRESULT(0)
        }
        _ => hwnd.def_proc(msg, wp, lp),
    }
}
