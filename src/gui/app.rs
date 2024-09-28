use anyhow::Result;
use windows::{core::PCSTR, Win32::Foundation::HWND};

use crate::config::Config;

pub trait App: Sized {
    fn new(config: Config, hwnd: HWND) -> Result<Self>;
    fn on_close(&mut self);
    fn on_move(&mut self, x: i32, y: i32);
    fn on_sized(&mut self, cx: i32, cy: i32);
    fn on_paint(&mut self);
    fn on_timer(&mut self);
    fn on_dpi_changed(&mut self, dpi: u32);
    fn on_menu(&mut self, id: u32, state: bool);
    fn menu_items(&self) -> Vec<MenuItem>;
}

pub enum MenuItem {
    Action { id: u32, text: PCSTR },
    CheckBox { id: u32, text: PCSTR, checked: bool },
    Radio { id: u32, text: PCSTR, checked: bool },
    Separator,
    SubMenu { text: PCSTR, items: Vec<MenuItem> },
}

#[macro_export]
macro_rules! action {
    ($id:expr, $text:literal) => {
        MenuItem::Action {
            id: $id,
            text: ::windows::core::s!($text),
        }
    };
}

#[macro_export]
macro_rules! checkbox {
    ($id:expr, $text:literal, $checked:expr $(,)?) => {
        MenuItem::CheckBox {
            id: $id,
            text: ::windows::core::s!($text),
            checked: $checked,
        }
    };
}

#[macro_export]
macro_rules! radio {
    ($id:expr, $text:literal, $checked:expr $(,)?) => {
        MenuItem::Radio {
            id: $id,
            text: ::windows::core::s!($text),
            checked: $checked,
        }
    };
}

#[macro_export]
macro_rules! separator {
    () => {
        MenuItem::Separator
    };
}

#[macro_export]
macro_rules! submenu {
    ($text:literal, $($item:expr),+ $(,)?) => {
        MenuItem::SubMenu {
            text: ::windows::core::s!($text),
            items: vec![$($item),+],
        }
    };
}
