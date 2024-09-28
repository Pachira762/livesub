use anyhow::{Context, Result};
use windows::{
    core::{s, PCSTR, PCWSTR},
    Win32::{
        Foundation::*,
        UI::{Input::KeyboardAndMouse::VK_ESCAPE, WindowsAndMessaging::*},
    },
};

use crate::{
    config::Config,
    gui::{
        utils::{self, Hwnd as _, Rect as _},
        window::{self, WindowClass},
    },
};

use super::{app::App, menu::ContextMenu, utils::Word, window::Window};

pub struct Viewer<T: App> {
    hwnd: HWND,
    app: Option<T>,
    menu: ContextMenu,
    show_menu: bool,
}

impl<T: App> Viewer<T> {
    pub fn create<'a>(config: Config) -> Result<&'a Self> {
        const CLASS_NAME: PCSTR = s!("livesub");

        WNDCLASSEXA::new()
            .set_style(CS_VREDRAW | CS_HREDRAW)
            .set_wndproc(window::wndproc::<Self>)
            .set_icon(utils::load_icon(Some(PCWSTR(1 as _))))
            .set_name(CLASS_NAME)
            .register()?;

        let hwnd = HWND::create(
            WS_EX_TOPMOST | WS_EX_NOREDIRECTIONBITMAP,
            CLASS_NAME,
            s!("livesub"),
            WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
            config.window_rect.x(),
            config.window_rect.y(),
            config.window_rect.width(),
            config.window_rect.height(),
            None,
            None,
            Some(Box::into_raw(Box::new(config)) as _),
        )?;

        hwnd.update();
        hwnd.show(SW_SHOW);

        Ok(unsafe {
            std::ptr::NonNull::new(hwnd.user_data() as *mut Self)
                .context("no binded viewer")?
                .as_ref()
        })
    }
}

impl<T: App> Window for Viewer<T> {
    fn new(hwnd: HWND, cs: &mut CREATESTRUCTA) -> Result<Box<Self>> {
        let config = unsafe { Box::from_raw(cs.lpCreateParams as *mut Config) };
        let app = T::new(config.as_ref().clone(), hwnd)?;
        let menu = ContextMenu::new(hwnd, &app.menu_items())?;

        Ok(Box::new(Self {
            hwnd,
            app: Some(app),
            menu,
            show_menu: false,
        }))
    }

    fn wndproc(&mut self, hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> Option<LRESULT> {
        match msg {
            WM_CLOSE => {
                hwnd.destroy();
                Some(LRESULT(0))
            }
            WM_KEYDOWN if wp.0 == VK_ESCAPE.0 as usize => {
                hwnd.destroy();
                Some(LRESULT(0))
            }
            WM_DESTROY => {
                if let Some(mut app) = self.app.take() {
                    app.on_close();
                }

                unsafe {
                    PostQuitMessage(0);
                }

                Some(LRESULT(0))
            }
            WM_MOVE => {
                if let Some(app) = &mut self.app {
                    app.on_move(lp.lo() as _, lp.hi() as _);
                }
                Some(LRESULT(0))
            }
            WM_SIZE => {
                if let Some(app) = &mut self.app {
                    app.on_sized(lp.lo() as _, lp.hi() as _);
                }
                Some(LRESULT(0))
            }
            WM_PAINT => {
                if let Some(app) = &mut self.app {
                    app.on_paint();
                }
                self.hwnd.validate_rect(None);
                Some(LRESULT(0))
            }
            WM_TIMER => {
                if let Some(app) = &mut self.app {
                    app.on_timer();
                }
                Some(LRESULT(0))
            }
            WM_DPICHANGED => {
                let dpi = wp.hi();
                if let Some(app) = &mut self.app {
                    app.on_dpi_changed(dpi);
                }

                let rect = unsafe { (lp.0 as *const RECT).as_ref().unwrap() };
                hwnd.set_pos(rect.x(), rect.y(), rect.width(), rect.height());

                Some(LRESULT(0))
            }
            WM_RBUTTONDOWN | WM_NCRBUTTONDOWN => {
                self.show_menu = true;
                Some(LRESULT(0))
            }
            WM_CONTEXTMENU => {
                self.show_menu = false;

                if let Some((id, state)) = self.menu.show() {
                    if let Some(app) = &mut self.app {
                        app.on_menu(id, state);
                    }
                }

                Some(LRESULT(0))
            }
            WM_NCCALCSIZE => Some(LRESULT(0)),
            WM_NCHITTEST => {
                if self.show_menu {
                    Some(LRESULT(HTCLIENT as _))
                } else {
                    Some(nc_hit_test(hwnd, lp.lo() as _, lp.hi() as _))
                }
            }
            _ => None,
        }
    }
}

fn nc_hit_test(hwnd: HWND, x: i32, y: i32) -> LRESULT {
    let RECT {
        left,
        top,
        right,
        bottom,
    } = hwnd.rect();

    let RECT {
        right: bx,
        bottom: by,
        ..
    } = *RECT::default().adjusted(WS_OVERLAPPEDWINDOW, FALSE, WINDOW_EX_STYLE(0), hwnd.dpi());

    let col = if x < left + bx {
        0
    } else if x < right - bx {
        1
    } else {
        2
    };

    let row = if y < top + by {
        0
    } else if y < bottom - by {
        1
    } else {
        2
    };

    match (col, row) {
        (0, 0) => LRESULT(HTTOPLEFT as _),
        (1, 0) => LRESULT(HTTOP as _),
        (2, 0) => LRESULT(HTTOPRIGHT as _),
        (0, 1) => LRESULT(HTLEFT as _),
        (1, 1) => LRESULT(HTCAPTION as _),
        (2, 1) => LRESULT(HTRIGHT as _),
        (0, 2) => LRESULT(HTBOTTOMLEFT as _),
        (1, 2) => LRESULT(HTBOTTOM as _),
        (2, 2) => LRESULT(HTBOTTOMRIGHT as _),
        _ => unreachable!(),
    }
}
