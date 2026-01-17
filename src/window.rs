use std::sync::{Arc, Mutex};

use anyhow::Result;
use windows::{
    Win32::{
        Foundation::*,
        Graphics::{
            Dwm::{DWMWA_USE_IMMERSIVE_DARK_MODE, DwmDefWindowProc, DwmSetWindowAttribute},
            Gdi::{UpdateWindow, ValidateRect},
        },
        System::LibraryLoader::{
            GetModuleHandleA, GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExA,
        },
        UI::{
            HiDpi::{AdjustWindowRectExForDpi, GetDpiForWindow},
            Input::KeyboardAndMouse::VK_ESCAPE,
            WindowsAndMessaging::*,
        },
    },
    core::*,
};

use crate::{
    config::*,
    controller::{Controller, WM_NEW_TRANSCRIPTION},
    renderer::Renderer,
};

const CMD_QUIT: u32 = 0x0001;
const CMD_CLEAR: u32 = 0x0002;
const CMD_MODEL_PARAKEET: u32 = 0x0101;
const CMD_MODEL_REAZONSPEECH: u32 = 0x0102;
const CMD_FONT_SIZE_SMALL: u32 = 0x0201;
const CMD_FONT_SIZE_MEDIUM: u32 = 0x0202;
const CMD_FONT_SIZE_LARGE: u32 = 0x0203;
const CMD_FONT_STYLE_BOLD: u32 = 0x0301;
const CMD_FONT_STYLE_ITALIC: u32 = 0x0302;
const CMD_BG_NONE: u32 = 0x0501;
const CMD_BG_LIGHT: u32 = 0x0502;
const CMD_BG_DARK: u32 = 0x0503;

pub struct Window {
    hwnd: HWND,
    menu: Menu,
    config: Box<Config>,
    controller: Controller,
    text: Arc<Mutex<String>>,
    renderer: Renderer,
}

impl Window {
    pub fn create(config: Box<Config>) -> Result<()> {
        unsafe {
            let hwnd = {
                const CLASSNAME: PCSTR = s!("Livesub");

                let instance: HINSTANCE = GetModuleHandleA(None)?.into();
                let wc = WNDCLASSEXA {
                    cbSize: std::mem::size_of::<WNDCLASSEXA>() as _,
                    lpfnWndProc: Some(wndproc),
                    hInstance: instance,
                    hIcon: LoadIconW(Some(instance), PCWSTR(1 as _))?,
                    hCursor: LoadCursorW(None, IDC_ARROW)?,
                    lpszClassName: CLASSNAME,
                    ..Default::default()
                };
                if RegisterClassExA(&wc) == 0 {
                    anyhow::bail!(windows::core::Error::from_thread())
                }

                CreateWindowExA(
                    WS_EX_TOPMOST | WS_EX_NOREDIRECTIONBITMAP,
                    CLASSNAME,
                    s!("Libsub"),
                    WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
                    config.x,
                    config.y,
                    config.width as _,
                    config.height as _,
                    None,
                    None,
                    Some(instance),
                    Some(Box::into_raw(config) as _),
                )?
            };

            _ = UpdateWindow(hwnd);
            _ = ShowWindow(hwnd, SW_SHOW);

            Ok(())
        }
    }

    fn new(hwnd: HWND, config: Box<Config>) -> Result<Self> {
        let menu = Menu::new(&config)?;
        let text = Arc::new(Mutex::new(String::new()));
        let controller = Controller::new(hwnd, Arc::clone(&text), &config.model_name)?;
        let background = match config.background.as_str() {
            "light" => [0.0, 0.0, 0.0, 0.5],
            "dark" => [0.0, 0.0, 0.0, 0.8],
            _ => [0.0, 0.0, 0.0, 0.0],
        };
        let renderer = Renderer::new(
            hwnd,
            &config.font_name,
            config.font_size as f32,
            config.font_style_bold,
            config.font_style_italic,
            background,
        )?;

        Ok(Self {
            hwnd,
            config,
            menu,
            controller,
            text,
            renderer,
        })
    }

    fn init(&mut self) -> Result<()> {
        unsafe {
            {
                #[allow(unused)]
                #[repr(C)]
                enum PreferredAppMode {
                    Default,
                    AllowDark,
                    ForceDark,
                    ForceLight,
                    Max,
                }
                type FnSetPreferredAppMode =
                    extern "system" fn(PreferredAppMode) -> PreferredAppMode;
                if let Ok(lib) =
                    LoadLibraryExA(s!("uxtheme.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32)
                {
                    if let Some(addr) = GetProcAddress(lib, PCSTR(135 as *const u8)) {
                        std::mem::transmute::<
                            unsafe extern "system" fn() -> isize,
                            FnSetPreferredAppMode,
                        >(addr)(PreferredAppMode::AllowDark);
                    }
                    FreeLibrary(lib)?;
                }
            }

            let value = BOOL(1);
            DwmSetWindowAttribute(
                self.hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                &raw const value as _,
                std::mem::size_of_val(&value) as _,
            )?;
            Ok(())
        }
    }

    fn close(&mut self) {
        unsafe {
            self.controller.shutdown();
            self.config.save();
            _ = DestroyWindow(self.hwnd);
        }
    }

    fn handle_menu(&mut self, id: u32) -> Result<()> {
        match id {
            CMD_QUIT => {
                self.close();
            }
            CMD_CLEAR => {
                self.controller.clear()?;
                self.renderer.set_text(String::new());
            }
            CMD_MODEL_PARAKEET => {
                self.config.model_name = "parakeet".into();
                self.controller.set_model("parakeet")?;
                self.renderer.set_text("Loading Parakeet...".into());
                self.renderer.render()?;
                self.menu.check_radio(
                    CMD_MODEL_PARAKEET,
                    CMD_MODEL_REAZONSPEECH,
                    CMD_MODEL_PARAKEET,
                )?;
            }
            CMD_MODEL_REAZONSPEECH => {
                self.config.model_name = "reazonspeech".into();
                self.controller.set_model("reazonspeech")?;
                self.renderer.set_text("Loading ReazonSpeech...".into());
                self.renderer.render()?;
                self.menu.check_radio(
                    CMD_MODEL_PARAKEET,
                    CMD_MODEL_REAZONSPEECH,
                    CMD_MODEL_REAZONSPEECH,
                )?;
            }
            CMD_FONT_SIZE_SMALL => {
                self.config.font_size = FONT_SIZE_SMALL;
                self.renderer.set_font_size(self.config.font_size as _);
                self.renderer.render()?;
                self.menu.check_radio(
                    CMD_FONT_SIZE_SMALL,
                    CMD_FONT_SIZE_LARGE,
                    CMD_FONT_SIZE_SMALL,
                )?;
            }
            CMD_FONT_SIZE_MEDIUM => {
                self.config.font_size = FONT_SIZE_MEDIUM;
                self.renderer.set_font_size(self.config.font_size as _);
                self.renderer.render()?;
                self.menu.check_radio(
                    CMD_FONT_SIZE_SMALL,
                    CMD_FONT_SIZE_LARGE,
                    CMD_FONT_SIZE_MEDIUM,
                )?;
            }
            CMD_FONT_SIZE_LARGE => {
                self.config.font_size = FONT_SIZE_LARGE;
                self.renderer.set_font_size(self.config.font_size as _);
                self.renderer.render()?;
                self.menu.check_radio(
                    CMD_FONT_SIZE_SMALL,
                    CMD_FONT_SIZE_LARGE,
                    CMD_FONT_SIZE_LARGE,
                )?;
            }
            CMD_FONT_STYLE_BOLD => {
                let bold = !self.menu.is_checked(CMD_FONT_STYLE_BOLD)?;
                self.config.font_style_bold = bold;
                self.renderer.set_font_style_bold(bold);
                self.renderer.render()?;
                self.menu.check(CMD_FONT_STYLE_BOLD, bold)?;
            }
            CMD_FONT_STYLE_ITALIC => {
                let italic = !self.menu.is_checked(CMD_FONT_STYLE_ITALIC)?;
                self.config.font_style_italic = italic;
                self.renderer.set_font_style_italic(italic);
                self.renderer.render()?;
                self.menu.check(CMD_FONT_STYLE_ITALIC, italic)?;
            }
            CMD_BG_NONE => {
                self.config.background = "none".into();
                self.renderer.set_background(0.0, 0.0, 0.0, 0.0);
                self.renderer.render()?;
                self.menu
                    .check_radio(CMD_BG_NONE, CMD_BG_DARK, CMD_BG_NONE)?;
            }
            CMD_BG_LIGHT => {
                self.config.background = "light".into();
                self.renderer.set_background(0.0, 0.0, 0.0, 0.5);
                self.renderer.render()?;
                self.menu
                    .check_radio(CMD_BG_NONE, CMD_BG_DARK, CMD_BG_LIGHT)?;
            }
            CMD_BG_DARK => {
                self.config.background = "dark".into();
                self.renderer.set_background(0.0, 0.0, 0.0, 0.8);
                self.renderer.render()?;
                self.menu
                    .check_radio(CMD_BG_NONE, CMD_BG_DARK, CMD_BG_DARK)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn nchittest(&mut self, hwnd: HWND) -> Result<LRESULT> {
        let RECT {
            left,
            top,
            right,
            bottom,
        } = unsafe {
            let mut rect = RECT::default();
            _ = GetWindowRect(hwnd, &mut rect);
            rect
        };

        let RECT {
            right: margin_x,
            bottom: margin_y,
            ..
        } = unsafe {
            let mut rect = RECT::default();
            _ = AdjustWindowRectExForDpi(
                &mut rect,
                WS_OVERLAPPEDWINDOW,
                false,
                WINDOW_EX_STYLE(0),
                GetDpiForWindow(hwnd),
            );
            rect
        };

        let POINT { x, y } = unsafe {
            let mut p = POINT::default();
            _ = GetCursorPos(&mut p);
            p
        };

        let col = if x < left + margin_x {
            0
        } else if x < right - margin_x {
            1
        } else {
            2
        };

        let row = if y < top + margin_y {
            0
        } else if y < bottom - margin_y {
            1
        } else {
            2
        };

        match (col, row) {
            (0, 0) => Ok(LRESULT(HTTOPLEFT as _)),
            (1, 0) => Ok(LRESULT(HTTOP as _)),
            (2, 0) => Ok(LRESULT(HTTOPRIGHT as _)),
            (0, 1) => Ok(LRESULT(HTLEFT as _)),
            (1, 1) => Ok(LRESULT(HTCAPTION as _)),
            (2, 1) => Ok(LRESULT(HTRIGHT as _)),
            (0, 2) => Ok(LRESULT(HTBOTTOMLEFT as _)),
            (1, 2) => Ok(LRESULT(HTBOTTOM as _)),
            (2, 2) => Ok(LRESULT(HTBOTTOMRIGHT as _)),
            _ => unreachable!(),
        }
    }

    fn handle_message(
        &mut self,
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> Result<LRESULT> {
        unsafe {
            match msg {
                WM_CREATE => {
                    self.init()?;
                    Ok(LRESULT(0))
                }
                WM_CLOSE => {
                    self.close();
                    Ok(LRESULT(0))
                }
                WM_KEYDOWN if wparam.0 == VK_ESCAPE.0 as usize => {
                    self.close();
                    Ok(LRESULT(0))
                }
                WM_DESTROY => {
                    PostQuitMessage(0);
                    Ok(LRESULT(0))
                }
                WM_WINDOWPOSCHANGED => {
                    let mut rect = RECT::default();
                    GetWindowRect(hwnd, &mut rect)?;

                    self.config.x = rect.left;
                    self.config.y = rect.top;
                    self.config.width = rect.right - rect.left;
                    self.config.height = rect.bottom - rect.top;

                    Ok(DefWindowProcA(hwnd, msg, wparam, lparam))
                }
                WM_SIZE => {
                    let width = lparam.0 as u32 & 0xffff;
                    let height = (lparam.0 as u32 >> 16) & 0xffff;
                    self.renderer.resize(width, height)?;
                    Ok(LRESULT(0))
                }
                WM_PAINT => {
                    self.renderer.render()?;
                    _ = ValidateRect(Some(hwnd), None);
                    Ok(LRESULT(0))
                }
                WM_RBUTTONUP | WM_NCRBUTTONUP | WM_CONTEXTMENU => {
                    if let Some(id) = self.menu.show(hwnd)? {
                        self.handle_menu(id)?;
                    }
                    Ok(LRESULT(0))
                }
                WM_NCCALCSIZE => Ok(LRESULT(0)),
                WM_NCHITTEST => self.nchittest(hwnd),
                WM_NEW_TRANSCRIPTION => {
                    let text = match self.text.lock() {
                        Ok(lock) => lock.clone(),
                        _ => String::from("Lock failed"),
                    };
                    self.renderer.set_text(text);
                    self.renderer.render()?;
                    Ok(LRESULT(0))
                }
                _ => Ok(DefWindowProcA(hwnd, msg, wparam, lparam)),
            }
        }
    }

    fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> Result<LRESULT> {
        unsafe {
            if msg == WM_NCCREATE {
                let cs = lparam.0 as *mut CREATESTRUCTA;
                let config = Box::from_raw((*cs).lpCreateParams as *mut Config);
                let window = Box::leak(Box::new(Self::new(hwnd, config)?));
                SetWindowLongPtrA(hwnd, GWLP_USERDATA, window as *mut _ as _);
            } else {
                let this = GetWindowLongPtrA(hwnd, GWLP_USERDATA) as *mut Self;
                if !this.is_null() {
                    return (*this).handle_message(hwnd, msg, wparam, lparam);
                }
            }
            Ok(DefWindowProcA(hwnd, msg, wparam, lparam))
        }
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    unsafe {
        let mut result = LRESULT::default();
        if DwmDefWindowProc(hwnd, msg, wparam, lparam, &mut result).as_bool() {
            return result;
        }

        match Window::wndproc(hwnd, msg, wparam, lparam) {
            Ok(ret) => ret,
            Err(e) => {
                eprintln!("{e:?}");
                PostQuitMessage(-1);
                LRESULT(0)
            }
        }
    }
}

struct Menu {
    hmenu: HMENU,
}

impl Menu {
    fn new(config: &Config) -> Result<Self> {
        unsafe {
            enum MenuItem {
                Action(u32, PCSTR),
                Checkbox(u32, PCSTR, bool),
                RadioGroup(Vec<(u32, PCSTR, bool)>),
                Submenu(PCSTR),
                PopSubmenu,
                Separator,
            }
            use MenuItem::*;

            let items = &[
                Action(CMD_CLEAR, s!("Clear")),
                Separator,
                Submenu(s!("Model")),
                RadioGroup(vec![
                    (
                        CMD_MODEL_PARAKEET,
                        s!("Parakeet"),
                        config.model_name == "parakeet",
                    ),
                    (
                        CMD_MODEL_REAZONSPEECH,
                        s!("ReazonSpeech"),
                        config.model_name == "reazonspeech",
                    ),
                ]),
                PopSubmenu,
                Submenu(s!("Background")),
                RadioGroup(vec![
                    (CMD_BG_NONE, s!("None"), config.background == "none"),
                    (CMD_BG_LIGHT, s!("Light"), config.background == "light"),
                    (CMD_BG_DARK, s!("Dark"), config.background == "dark"),
                ]),
                PopSubmenu,
                Submenu(s!("Font Size")),
                RadioGroup(vec![
                    (
                        CMD_FONT_SIZE_SMALL,
                        s!("Small"),
                        config.font_size == FONT_SIZE_SMALL,
                    ),
                    (
                        CMD_FONT_SIZE_MEDIUM,
                        s!("Medium"),
                        config.font_size == FONT_SIZE_MEDIUM,
                    ),
                    (
                        CMD_FONT_SIZE_LARGE,
                        s!("Large"),
                        config.font_size == FONT_SIZE_LARGE,
                    ),
                ]),
                PopSubmenu,
                Submenu(s!("Font Style")),
                Checkbox(CMD_FONT_STYLE_BOLD, s!("Bold"), config.font_style_bold),
                Checkbox(
                    CMD_FONT_STYLE_ITALIC,
                    s!("Italic"),
                    config.font_style_italic,
                ),
                PopSubmenu,
                Separator,
                Action(CMD_QUIT, s!("Quit")),
            ];

            let hmenu = CreatePopupMenu()?;
            let mut submenus = vec![hmenu];

            for item in items {
                match item {
                    &Action(id, label) => {
                        let hmenu = *submenus.last().unwrap();
                        let mi = MENUITEMINFOA {
                            cbSize: std::mem::size_of::<MENUITEMINFOA>() as u32,
                            fMask: MIIM_ID | MIIM_STRING,
                            fType: MFT_STRING,
                            wID: id,
                            dwTypeData: PSTR(label.0 as _),
                            ..Default::default()
                        };
                        InsertMenuItemA(hmenu, u32::MAX, true, &mi)?;
                    }
                    &Checkbox(id, label, checked) => {
                        let hmenu = *submenus.last().unwrap();
                        let state = if checked { MFS_CHECKED } else { MFS_UNCHECKED };
                        let mi = MENUITEMINFOA {
                            cbSize: std::mem::size_of::<MENUITEMINFOA>() as u32,
                            fMask: MIIM_ID | MIIM_STRING | MIIM_STATE,
                            fType: MFT_STRING,
                            wID: id,
                            dwTypeData: PSTR(label.0 as _),
                            fState: state,
                            ..Default::default()
                        };
                        InsertMenuItemA(hmenu, u32::MAX, true, &mi)?;
                    }
                    RadioGroup(items) => {
                        let hmenu = *submenus.last().unwrap();
                        for &(id, label, checked) in items {
                            let state = if checked { MFS_CHECKED } else { MFS_UNCHECKED };
                            let mi = MENUITEMINFOA {
                                cbSize: std::mem::size_of::<MENUITEMINFOA>() as u32,
                                fMask: MIIM_ID | MIIM_STRING,
                                fType: MFT_STRING | MFT_RADIOCHECK,
                                wID: id,
                                dwTypeData: PSTR(label.0 as _),
                                fState: state,
                                ..Default::default()
                            };
                            InsertMenuItemA(hmenu, u32::MAX, true, &mi)?;
                        }

                        let first = items.first().unwrap().0;
                        let last = items.last().unwrap().0;
                        let check = items
                            .iter()
                            .find(|item| item.2)
                            .map(|item| item.0)
                            .unwrap_or(first);
                        CheckMenuRadioItem(hmenu, first, last, check, MF_BYCOMMAND.0)?;
                    }
                    &Submenu(label) => {
                        let hmenu = *submenus.last().unwrap();
                        let submenu = CreatePopupMenu()?;
                        let mi = MENUITEMINFOA {
                            cbSize: std::mem::size_of::<MENUITEMINFOA>() as u32,
                            fMask: MIIM_STRING | MIIM_SUBMENU,
                            fType: MFT_STRING,
                            dwTypeData: PSTR(label.0 as _),
                            hSubMenu: submenu,
                            ..Default::default()
                        };
                        InsertMenuItemA(hmenu, u32::MAX, true, &mi)?;
                        submenus.push(submenu);
                    }
                    PopSubmenu => {
                        _ = submenus.pop();
                    }
                    Separator => {
                        let hmenu = *submenus.last().unwrap();
                        let mi = MENUITEMINFOA {
                            cbSize: std::mem::size_of::<MENUITEMINFOA>() as u32,
                            fType: MFT_SEPARATOR,
                            ..Default::default()
                        };
                        InsertMenuItemA(hmenu, u32::MAX, true, &mi)?;
                    }
                }
            }

            Ok(Self { hmenu })
        }
    }

    fn show(&mut self, hwnd: HWND) -> Result<Option<u32>> {
        unsafe {
            let mut cursor = POINT::default();
            GetCursorPos(&mut cursor)?;

            let id = TrackPopupMenu(
                self.hmenu,
                TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RETURNCMD,
                cursor.x,
                cursor.y,
                None,
                hwnd,
                None,
            );

            if id == BOOL(0) {
                Ok(None)
            } else {
                Ok(Some(id.0 as u32))
            }
        }
    }

    fn check(&self, id: u32, checked: bool) -> Result<()> {
        unsafe {
            let state = if checked { MFS_CHECKED } else { MFS_UNCHECKED };
            let mi = MENUITEMINFOA {
                cbSize: std::mem::size_of::<MENUITEMINFOA>() as _,
                fMask: MIIM_STATE,
                fState: state,
                ..Default::default()
            };
            SetMenuItemInfoA(self.hmenu, id, false, &mi)?;
            Ok(())
        }
    }

    fn check_radio(&self, first: u32, last: u32, check: u32) -> Result<()> {
        unsafe {
            CheckMenuRadioItem(self.hmenu, first, last, check, MF_BYCOMMAND.0)?;
            Ok(())
        }
    }

    fn is_checked(&self, id: u32) -> Result<bool> {
        unsafe {
            let mut mi = MENUITEMINFOA {
                cbSize: std::mem::size_of::<MENUITEMINFOA>() as _,
                fMask: MIIM_STATE,
                ..Default::default()
            };
            GetMenuItemInfoA(self.hmenu, id, false, &mut mi)?;
            println!("GetMenuItemInfo {mi:?}");
            Ok((mi.fState & MFS_CHECKED) != MENU_ITEM_STATE(0))
        }
    }
}
