#![allow(unused, non_snake_case, clippy::too_many_arguments)]

use anyhow::{Error as E, Result};
use windows::{
    core::{s, Param, PCSTR, PCWSTR},
    Win32::{
        Foundation::*,
        Graphics::{
            Dwm::{
                DwmDefWindowProc, DwmEnableBlurBehindWindow, DwmExtendFrameIntoClientArea,
                DwmGetWindowAttribute, DwmSetWindowAttribute, DWMWA_CAPTION_BUTTON_BOUNDS,
                DWMWINDOWATTRIBUTE, DWM_BB_ENABLE, DWM_BLURBEHIND,
            },
            Gdi::{UpdateWindow, ValidateRect},
        },
        System::LibraryLoader::{
            GetModuleHandleA, GetProcAddress, LoadLibraryExA, LOAD_LIBRARY_SEARCH_SYSTEM32,
        },
        UI::{
            Controls::*,
            HiDpi::{AdjustWindowRectExForDpi, GetDpiForWindow},
            WindowsAndMessaging::*,
        },
    },
};

#[macro_export]
macro_rules! LOWORD {
    ($dw:expr) => {
        ($dw & 0xffff) as u32
    };
}

#[macro_export]
macro_rules! HIWORD {
    ($dw:expr) => {
        (($dw >> 16) & 0xffff) as u32
    };
}

#[macro_export]
macro_rules! GET_X_LPARAM {
    ($lp:ident) => {
        (($lp.0 & 0xffff) as i16) as i32
    };
}

#[macro_export]
macro_rules! GET_Y_LPARAM {
    ($lp:ident) => {
        ((($lp.0 >> 16) & 0xffff) as i16) as i32
    };
}

#[macro_export]
macro_rules! GET_WHEEL_DELTA_WPARAM {
    ($wp:ident) => {
        ((($wp.0 >> 16) & 0xffff) as i16) as i32
    };
}

pub trait Wstr {
    fn c_wstr(&self) -> Vec<u16>;
}

impl<'a> Wstr for &'a str {
    fn c_wstr(&self) -> Vec<u16> {
        self.encode_utf16().chain([0]).collect()
    }
}

impl Wstr for String {
    fn c_wstr(&self) -> Vec<u16> {
        self.as_str().c_wstr()
    }
}

pub trait Word: Sized {
    fn lo(self) -> u32;
    fn hi(self) -> u32;
}

impl Word for WPARAM {
    fn lo(self) -> u32 {
        LOWORD!(self.0)
    }

    fn hi(self) -> u32 {
        HIWORD!(self.0)
    }
}

impl Word for LPARAM {
    fn lo(self) -> u32 {
        LOWORD!(self.0)
    }

    fn hi(self) -> u32 {
        HIWORD!(self.0)
    }
}

pub trait Rect: Sized {
    fn new(x: i32, y: i32, width: i32, height: i32) -> Self;
    fn inner(&self, x: i32, y: i32) -> Self;
    fn adjusted(
        &mut self,
        style: WINDOW_STYLE,
        bmenu: BOOL,
        ex_style: WINDOW_EX_STYLE,
        dpi: u32,
    ) -> &mut Self;

    fn x(&self) -> i32;
    fn y(&self) -> i32;
    fn width(&self) -> i32;
    fn height(&self) -> i32;
    fn size(&self) -> (i32, i32);

    fn set_x(&mut self, x: i32);
    fn set_y(&mut self, y: i32);
    fn set_width(&mut self, width: i32);
    fn set_height(&mut self, height: i32);
    fn set_size(&mut self, width: i32, height: i32);

    fn is_in(&self, x: i32, y: i32) -> bool;
}

impl Rect for RECT {
    fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            left: x,
            top: y,
            right: x + width,
            bottom: y + height,
        }
    }

    fn inner(&self, x: i32, y: i32) -> RECT {
        RECT {
            left: self.left + x,
            top: self.top + y,
            right: self.right - x,
            bottom: self.bottom - y,
        }
    }

    fn adjusted(
        &mut self,
        style: WINDOW_STYLE,
        bmenu: BOOL,
        ex_style: WINDOW_EX_STYLE,
        dpi: u32,
    ) -> &mut Self {
        unsafe {
            AdjustWindowRectExForDpi(self, style, bmenu, ex_style, dpi)
                .expect("failed to AdjustWindowRectEx.");
            self
        }
    }

    fn x(&self) -> i32 {
        self.left
    }
    fn y(&self) -> i32 {
        self.top
    }

    fn width(&self) -> i32 {
        self.right - self.left
    }

    fn height(&self) -> i32 {
        self.bottom - self.top
    }

    fn size(&self) -> (i32, i32) {
        (self.right - self.left, self.bottom - self.top)
    }

    fn set_x(&mut self, x: i32) {
        let width = self.width();
        self.left = x;
        self.right = x + width;
    }

    fn set_y(&mut self, y: i32) {
        let height = self.height();
        self.top = y;
        self.bottom = y + height;
    }

    fn set_width(&mut self, width: i32) {
        self.right = self.left + width;
    }

    fn set_height(&mut self, height: i32) {
        self.bottom = self.top + height;
    }

    fn set_size(&mut self, width: i32, height: i32) {
        self.right = self.left + width;
        self.bottom = self.top + height;
    }

    fn is_in(&self, x: i32, y: i32) -> bool {
        self.left <= x && x < self.right && self.top <= y && y < self.bottom
    }
}

pub trait Hinstance {
    fn get() -> HINSTANCE {
        unsafe { HINSTANCE(GetModuleHandleA(None).unwrap().0) }
    }
}

impl Hinstance for HINSTANCE {}

pub trait Hwnd: Copy + Into<HWND> {
    fn create<P0, P1>(
        exstyle: WINDOW_EX_STYLE,
        classname: PCSTR,
        windowname: PCSTR,
        style: WINDOW_STYLE,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        parent: P0,
        menu: P1,
        param: Option<*const std::ffi::c_void>,
    ) -> Result<HWND>
    where
        P0: Param<HWND>,
        P1: Param<HMENU>,
    {
        unsafe {
            CreateWindowExA(
                exstyle,
                classname,
                windowname,
                style,
                x,
                y,
                width,
                height,
                parent,
                menu,
                HINSTANCE::get(),
                param,
            )
            .map_err(anyhow::Error::msg)
        }
    }

    fn from_lparam(lp: LPARAM) -> HWND {
        HWND(lp.0 as _)
    }

    fn rect(self) -> RECT {
        unsafe {
            let mut rc = RECT::default();
            _ = GetWindowRect(self.into(), &mut rc);
            rc
        }
    }

    fn size(self) -> (u32, u32) {
        let (width, height) = self.rect().size();
        (width as _, height as _)
    }

    fn client_rect(self) -> RECT {
        unsafe {
            let mut rc = RECT::default();
            _ = GetWindowRect(self.into(), &mut rc);
            rc
        }
    }

    fn client_size(self) -> (u32, u32) {
        let (width, height) = self.client_rect().size();
        (width as _, height as _)
    }

    fn set_pos(self, x: i32, y: i32, width: i32, height: i32) {
        unsafe {
            SetWindowPos(
                self.into(),
                None,
                x,
                y,
                width,
                height,
                SWP_NOACTIVATE | SWP_NOZORDER,
            );
        }
    }

    fn style(self) -> WINDOW_STYLE {
        unsafe { WINDOW_STYLE(GetWindowLongA(self.into(), GWL_STYLE) as _) }
    }

    fn set_style(self, style: WINDOW_STYLE) {
        unsafe {
            SetWindowLongA(self.into(), GWL_STYLE, style.0 as _);
        }
    }

    fn ex_style(self) -> WINDOW_EX_STYLE {
        unsafe { WINDOW_EX_STYLE(GetWindowLongA(self.into(), GWL_EXSTYLE) as _) }
    }

    fn set_ex_style(self, ex_style: WINDOW_EX_STYLE) {
        unsafe {
            SetWindowLongA(self.into(), GWL_EXSTYLE, ex_style.0 as _);
        }
    }

    fn text(self) -> String {
        unsafe {
            let len = GetWindowTextLengthA(self.into());
            if len > 0 {
                let mut buf = vec![0; len as usize + 1];
                GetWindowTextA(self.into(), &mut buf);
                buf.pop();
                String::from_utf8_unchecked(buf)
            } else {
                String::new()
            }
        }
    }

    fn user_data(self) -> isize {
        unsafe { GetWindowLongPtrA(self.into(), GWLP_USERDATA) }
    }

    fn set_user_data(self, data: isize) {
        unsafe {
            SetWindowLongPtrA(self.into(), GWLP_USERDATA, data);
        }
    }

    fn window(self, cmd: GET_WINDOW_CMD) -> HWND {
        unsafe { GetWindow(self.into(), cmd).unwrap_or_default() }
    }

    fn owner(self) -> HWND {
        self.window(GW_OWNER)
    }

    fn parent(self) -> HWND {
        unsafe { GetParent(self.into()).unwrap_or_default() }
    }

    fn menu(self) -> HMENU {
        unsafe { GetMenu(self.into()) }
    }

    fn titlebar_info_ex(self) -> TITLEBARINFOEX {
        let mut info = TITLEBARINFOEX {
            cbSize: std::mem::size_of::<TITLEBARINFOEX>() as u32,
            ..Default::default()
        };

        _ = self.send_message(
            WM_GETTITLEBARINFOEX,
            WPARAM::default(),
            LPARAM(&mut info as *mut _ as _),
        );

        info
    }

    fn send_message(self, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        unsafe { SendMessageA(self.into(), msg, wp, lp) }
    }

    fn post_message(self, msg: u32, wp: WPARAM, lp: LPARAM) {
        unsafe {
            PostMessageA(self.into(), msg, wp, lp);
        }
    }

    fn def_proc(self, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        unsafe { DefWindowProcA(self.into(), msg, wp, lp) }
    }

    fn destroy(self) {
        unsafe {
            _ = DestroyWindow(self.into());
        }
    }

    fn update(self) {
        unsafe {
            UpdateWindow(self.into());
        }
    }

    fn validate_rect(self, rect: Option<*const RECT>) {
        unsafe {
            _ = ValidateRect(self.into(), rect);
        }
    }

    fn show(self, cmd: SHOW_WINDOW_CMD) {
        unsafe {
            ShowWindow(self.into(), cmd);
        }
    }

    fn set_timer(self, id: usize, elapse: u32) -> usize {
        unsafe { SetTimer(self.into(), id, elapse, None) }
    }

    fn scroll_info(
        self,
        bar: SCROLLBAR_CONSTANTS,
        range: bool,
        page: bool,
        pos: bool,
        track: bool,
    ) -> SCROLLINFO {
        unsafe {
            let mut info = SCROLLINFO::default();
            info.cbSize = std::mem::size_of_val(&info) as u32;
            if range {
                info.fMask |= SIF_RANGE;
            }
            if page {
                info.fMask |= SIF_PAGE;
            }
            if pos {
                info.fMask |= SIF_POS;
            }
            if track {
                info.fMask |= SIF_TRACKPOS;
            }
            GetScrollInfo(self.into(), bar, &mut info);
            info
        }
    }

    fn set_scroll_info(
        self,
        bar: SCROLLBAR_CONSTANTS,
        range: Option<(i32, i32)>,
        page: Option<u32>,
        pos: Option<i32>,
        track: Option<i32>,
    ) -> i32 {
        unsafe {
            let mut info = SCROLLINFO {
                cbSize: std::mem::size_of::<SCROLLINFO>() as u32,
                ..Default::default()
            };

            if let Some((min, max)) = range {
                info.fMask |= SIF_RANGE;
                info.nMin = min;
                info.nMax = max;
            }
            if let Some(page) = page {
                info.fMask |= SIF_PAGE;
                info.nPage = page;
            }
            if let Some(pos) = pos {
                info.fMask |= SIF_POS;
                info.nPos = pos;
            }
            if let Some(track) = track {
                info.fMask |= SIF_TRACKPOS;
                info.nTrackPos = track;
            }

            SetScrollInfo(self.into(), bar, &info, TRUE)
        }
    }

    fn scroll(self, dx: i32, dy: i32) {
        unsafe {
            ScrollWindowEx(
                self.into(),
                dx,
                dy,
                None,
                None,
                None,
                None,
                SW_ERASE | SW_INVALIDATE | SW_SCROLLCHILDREN,
            );
        }
    }

    fn dpi(self) -> u32 {
        unsafe { GetDpiForWindow(self.into()) }
    }

    fn set_theme(self, theme: PCWSTR) {
        unsafe {
            SetWindowTheme(self.into(), theme, None).expect("SetWindowTheme failed.");
        }
    }

    fn set_display_affinity(self, affinity: WINDOW_DISPLAY_AFFINITY) {
        unsafe {
            SetWindowDisplayAffinity(self.into(), affinity);
        }
    }

    fn dwm_extend_frame(self, left: i32, right: i32, top: i32, bottom: i32) -> Result<()> {
        unsafe {
            DwmExtendFrameIntoClientArea(
                self.into(),
                &MARGINS {
                    cxLeftWidth: left,
                    cxRightWidth: right,
                    cyTopHeight: top,
                    cyBottomHeight: bottom,
                },
            )
            .map_err(E::msg)
        }
    }

    fn dwm_enable_blur_behind(self, enable: bool) {
        unsafe {
            DwmEnableBlurBehindWindow(
                self.into(),
                &DWM_BLURBEHIND {
                    dwFlags: DWM_BB_ENABLE,
                    fEnable: enable.into(),
                    ..Default::default()
                },
            );
        }
    }

    fn dwm_attribute<T: Default>(self, attr: DWMWINDOWATTRIBUTE) -> T {
        unsafe {
            let mut value = T::default();
            DwmGetWindowAttribute(
                self.into(),
                attr,
                &mut value as *mut _ as _,
                std::mem::size_of::<T>() as u32,
            );
            value
        }
    }

    fn caption_button_bounds(self) -> RECT {
        self.dwm_attribute(DWMWA_CAPTION_BUTTON_BOUNDS)
    }

    fn dwm_set_attribute<T>(self, attr: DWMWINDOWATTRIBUTE, value: *const T) -> Result<()> {
        unsafe {
            DwmSetWindowAttribute(
                self.into(),
                attr,
                value as _,
                std::mem::size_of::<T>() as u32,
            )
            .map_err(E::msg)
        }
    }

    fn dwm_def_proc(self, msg: u32, wp: WPARAM, lp: LPARAM) -> Option<LRESULT> {
        unsafe {
            let mut result = LRESULT::default();
            if DwmDefWindowProc(self.into(), msg, wp, lp, &mut result).as_bool() {
                Some(result)
            } else {
                None
            }
        }
    }

    fn checkbox_checked(self) -> bool {
        self.send_message(BM_GETCHECK, WPARAM::default(), LPARAM::default())
            .0
            == BST_CHECKED.0 as isize
    }

    fn checkbox_set_check(self, checked: bool) {
        let state = if checked { BST_CHECKED } else { BST_UNCHECKED };
        self.send_message(BM_SETCHECK, WPARAM(state.0 as _), LPARAM::default());
    }

    fn trackbar_set_min_max(self, min: i32, max: i32) {
        self.send_message(TBM_SETRANGEMIN, WPARAM(0), LPARAM(min as _));
        self.send_message(TBM_SETRANGEMAX, WPARAM(1), LPARAM(max as _));
    }

    fn trackbar_set_pos(self, pos: i32) {
        self.send_message(TBM_SETPOS, WPARAM(1), LPARAM(pos as _));
    }

    fn trackbar_pos(self) -> i32 {
        const TBM_GETPOS: u32 = WM_USER;
        self.send_message(TBM_GETPOS, WPARAM(0), LPARAM(0)).0 as _
    }
}

impl Hwnd for HWND {}

pub fn quit(code: i32) {
    unsafe {
        PostQuitMessage(code);
    }
}

pub fn cursor_pos() -> (i32, i32) {
    unsafe {
        let mut point = POINT::default();
        GetCursorPos(&mut point);
        (point.x, point.y)
    }
}

pub fn system_metrics(index: SYSTEM_METRICS_INDEX) -> i32 {
    unsafe { GetSystemMetrics(index) }
}

pub fn load_icon(name: Option<PCWSTR>) -> HICON {
    let (instance, name) = match name {
        Some(name) => (HINSTANCE::get(), name),
        None => (HINSTANCE::default(), IDI_APPLICATION),
    };
    unsafe { LoadIconW(instance, name).unwrap() }
}

pub fn load_cursor(name: Option<PCWSTR>) -> HCURSOR {
    let (instance, name) = match name {
        Some(name) => (HINSTANCE::get(), name),
        None => (HINSTANCE::default(), IDI_APPLICATION),
    };
    unsafe { LoadCursorW(instance, name).unwrap() }
}
