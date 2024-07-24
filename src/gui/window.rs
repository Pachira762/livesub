use anyhow::Result;
use std::mem::{size_of_val, transmute};
use windows::{
    core::{s, PCWSTR},
    Win32::{
        Foundation::*,
        Graphics::{
            Dwm::{
                DwmDefWindowProc, DwmExtendFrameIntoClientArea, DwmSetWindowAttribute,
                DWMSBT_TRANSIENTWINDOW, DWMWA_NCRENDERING_POLICY, DWMWA_SYSTEMBACKDROP_TYPE,
                DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND,
            },
            Gdi::*,
        },
        System::LibraryLoader::GetModuleHandleA,
        UI::{
            Controls::MARGINS,
            HiDpi::{AdjustWindowRectExForDpi, GetDpiForWindow},
            Input::KeyboardAndMouse::VK_ESCAPE,
            WindowsAndMessaging::*,
        },
    },
};

use crate::app::Config;

use super::{macros::*, menu::Menu, MenuBuilder};

pub trait WinApp: Sized {
    type Command: From<u32>;

    fn new(config: Config, hwnd: HWND, menu: &mut MenuBuilder) -> Result<Self>;
    fn on_close(&mut self);
    fn on_sized(&mut self, cx: i32, cy: i32);
    fn on_paint(&mut self);
    fn on_timer(&mut self);
    fn on_menu(&mut self, id: Self::Command, state: Option<bool>);
    fn on_dpi_changed(&mut self, dpi: u32);
}

pub struct Window<T: WinApp> {
    app: Option<T>,
    menu: Menu,
    show_menu: bool,
}

impl<T: WinApp> Window<T> {
    pub fn new() -> Result<Self> {
        Ok(Self {
            app: None,
            menu: Menu::new()?,
            show_menu: false,
        })
    }

    fn bind_window(&mut self, config: Config, hwnd: HWND) -> Result<()> {
        let mut builder = self.menu.get_builder();
        self.app = Some(T::new(config, hwnd, &mut builder)?);
        Ok(())
    }

    fn wnd_proc(&mut self, hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        unsafe {
            match msg {
                WM_CLOSE => {
                    if let Some(app) = &mut self.app {
                        app.on_close();
                    }
                    _ = DestroyWindow(hwnd);
                    LRESULT::default()
                }
                WM_DESTROY => {
                    PostQuitMessage(0);
                    LRESULT::default()
                }
                WM_SIZE => {
                    if let Some(app) = &mut self.app {
                        app.on_sized(LOWORD(lp.0 as _), HIWORD(lp.0 as _));
                    }
                    LRESULT::default()
                }
                WM_PAINT => {
                    if let Some(app) = &mut self.app {
                        app.on_paint();
                    }
                    ValidateRect(hwnd, None);
                    LRESULT::default()
                }
                WM_TIMER => {
                    if let Some(app) = &mut self.app {
                        app.on_timer();
                    }
                    LRESULT::default()
                }
                WM_KEYDOWN => {
                    if wp.0 == VK_ESCAPE.0 as usize {
                        if let Some(app) = &mut self.app {
                            app.on_close();
                        }
                        _ = DestroyWindow(hwnd);
                    }
                    LRESULT::default()
                }
                WM_RBUTTONDOWN | WM_NCRBUTTONDOWN => {
                    self.show_menu = true;
                    DefWindowProcA(hwnd, msg, wp, lp)
                }
                WM_CONTEXTMENU => {
                    self.show_menu = false;

                    if let (Some((id, state)), Some(app)) = (self.menu.show(hwnd), &mut self.app) {
                        app.on_menu(id, state);
                    }

                    LRESULT::default()
                }
                WM_DPICHANGED => {
                    if let Some(app) = &mut self.app {
                        let dpi = LOWORD(wp.0) as u32;
                        app.on_dpi_changed(dpi);

                        let rc = *(lp.0 as *const RECT);
                        _ = SetWindowPos(
                            hwnd,
                            None,
                            rc.left,
                            rc.top,
                            rc.right - rc.left,
                            rc.bottom - rc.top,
                            SWP_NOACTIVATE | SWP_NOZORDER,
                        );
                    }
                    LRESULT::default()
                }
                WM_NCHITTEST => {
                    if self.show_menu {
                        DefWindowProcA(hwnd, msg, wp, lp)
                    } else {
                        nc_hit_test(hwnd)
                    }
                }
                _ => DefWindowProcA(hwnd, msg, wp, lp),
            }
        }
    }
}

pub fn run_app<T: WinApp>() -> Result<()> {
    unsafe {
        let instance = GetModuleHandleA(None)?;

        let window_class = s!("window");

        let wc = WNDCLASSA {
            hIcon: LoadIconW(instance, PCWSTR(1 as _))?,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            hInstance: instance.into(),
            lpszClassName: window_class,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wndproc::<T>),
            ..Default::default()
        };

        let atom = RegisterClassA(&wc);
        debug_assert!(atom != 0);

        let config = Config::from_ini();
        let mut window = Window::<T>::new()?;

        let hwnd = CreateWindowExA(
            WS_EX_TOPMOST | WS_EX_NOREDIRECTIONBITMAP,
            window_class,
            s!("livesub"),
            WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            config.width as i32,
            config.height as i32,
            None,
            None,
            instance,
            Some(&mut window as *mut _ as *const _),
        )?;
        window.bind_window(config, hwnd)?;

        UpdateWindow(hwnd);
        ShowWindow(hwnd, SW_SHOW);

        loop {
            let mut msg = MSG::default();

            match GetMessageA(&mut msg, None, 0, 0).0 {
                0 | -1 => break,
                _ => {
                    TranslateMessage(&msg);
                    DispatchMessageA(&msg);
                }
            }
        }

        Ok(())
    }
}

extern "system" fn wndproc<T: WinApp>(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    unsafe {
        let mut dwm_result = LRESULT(0);
        let dwm_proc: bool = DwmDefWindowProc(hwnd, msg, wp, lp, &mut dwm_result).into();

        match msg {
            WM_CREATE => {
                let mut rc = RECT::default();
                GetWindowRect(hwnd, &mut rc);
                SetWindowPos(
                    hwnd,
                    None,
                    rc.left,
                    rc.top,
                    rc.right - rc.left,
                    rc.bottom - rc.top,
                    SWP_FRAMECHANGED,
                );

                round_window_rect(hwnd);
                update_dwm(hwnd);

                let cs: &CREATESTRUCTA = transmute(lp);
                SetWindowLongPtrA(hwnd, GWLP_USERDATA, cs.lpCreateParams as _);
                LRESULT::default()
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                LRESULT(0)
            }
            WM_DWMNCRENDERINGCHANGED => {
                update_dwm(hwnd);
                LRESULT(0)
            }
            WM_NCCALCSIZE => nc_calc_size(hwnd, wp, lp),
            WM_NCHITTEST if dwm_proc => dwm_result,
            _ => {
                let ptr = GetWindowLongPtrA(hwnd, GWLP_USERDATA);
                let mut window = std::ptr::NonNull::<Window<T>>::new(ptr as _);

                if let Some(window) = &mut window {
                    window.as_mut().wnd_proc(hwnd, msg, wp, lp)
                } else {
                    DefWindowProcA(hwnd, msg, wp, lp)
                }
            }
        }
    }
}

fn round_window_rect(hwnd: HWND) {
    unsafe {
        let corner = DWMWCP_ROUND;
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            &corner as *const _ as _,
            std::mem::size_of_val(&corner) as u32,
        );
    }
}

fn nc_calc_size(hwnd: HWND, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if wp == WPARAM(1) {
        LRESULT(0)
    } else {
        unsafe { DefWindowProcA(hwnd, WM_NCCALCSIZE, wp, lp) }
    }
}

fn nc_hit_test(hwnd: HWND) -> LRESULT {
    enum Region {
        Outside,
        Caption,
        FrameA,
        Client,
        FrameB,
    }
    use Region::*;

    impl Region {
        fn detect(pos: i32, beg: i32, end: i32, frame: i32, caption: i32) -> Self {
            if pos < beg || pos >= end {
                Outside
            } else if pos < beg + frame {
                FrameA
            } else if pos >= end - frame {
                FrameB
            } else if pos < beg + caption || pos >= end - caption {
                Caption
            } else {
                Client
            }
        }
    }

    let (x, y) = unsafe {
        let mut p = POINT::default();
        _ = GetCursorPos(&mut p);
        (p.x, p.y)
    };

    let rect = unsafe {
        let mut rc = RECT::default();
        _ = GetWindowRect(hwnd, &mut rc);
        rc
    };

    let frame = unsafe {
        let dpi = GetDpiForWindow(hwnd);
        let mut frame = Default::default();
        _ = AdjustWindowRectExForDpi(
            &mut frame,
            WS_OVERLAPPEDWINDOW,
            FALSE,
            WINDOW_EX_STYLE(0),
            dpi,
        );
        frame
    };

    let rx = Region::detect(x, rect.left, rect.right, -frame.left, -frame.top - 1);
    let ry = Region::detect(y, rect.top, rect.bottom, -frame.left, -frame.top - 1);

    match (rx, ry) {
        (Outside, _) | (_, Outside) => LRESULT(HTNOWHERE as _),
        (Caption, _) | (_, Caption) => LRESULT(HTCAPTION as _),
        (FrameA, FrameA) => LRESULT(HTTOPLEFT as _),
        (Client, FrameA) => LRESULT(HTTOP as _),
        (FrameB, FrameA) => LRESULT(HTTOPRIGHT as _),
        (FrameA, Client) => LRESULT(HTLEFT as _),
        (Client, Client) => LRESULT(HTCAPTION as _),
        (FrameB, Client) => LRESULT(HTRIGHT as _),
        (FrameA, FrameB) => LRESULT(HTBOTTOMLEFT as _),
        (Client, FrameB) => LRESULT(HTBOTTOM as _),
        (FrameB, FrameB) => LRESULT(HTBOTTOMRIGHT as _),
    }
}

fn extend_frame(hwnd: HWND) {
    unsafe {
        DwmExtendFrameIntoClientArea(
            hwnd,
            &MARGINS {
                cxLeftWidth: -1,
                cxRightWidth: -1,
                cyTopHeight: -1,
                cyBottomHeight: -1,
            },
        )
        .expect("failed extend frame");
    }
}

fn update_dwm(hwnd: HWND) {
    unsafe {
        extend_frame(hwnd);

        let policy = BOOL(1);
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_NCRENDERING_POLICY,
            &policy as *const _ as _,
            size_of_val(&policy) as u32,
        )
        .expect("failed enable dwm policy");

        let darkmode = BOOL(1);
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &darkmode as *const _ as _,
            size_of_val(&darkmode) as u32,
        )
        .expect("failed enable darkmode");

        let attr = DWMSBT_TRANSIENTWINDOW;
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_SYSTEMBACKDROP_TYPE,
            &attr as *const _ as _,
            size_of_val(&attr) as u32,
        )
        .expect("failed system backdrop policy");
    }
}
