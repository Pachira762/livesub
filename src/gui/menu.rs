#![allow(unused)]

use std::collections::HashMap;

use anyhow::Result;
use windows::{
    core::{s, PCSTR, PSTR},
    Win32::{
        Foundation::{FreeLibrary, HWND},
        System::LibraryLoader::{GetProcAddress, LoadLibraryExA, LOAD_LIBRARY_SEARCH_SYSTEM32},
        UI::WindowsAndMessaging::*,
    },
};
use windows_core::BOOL;

use super::{
    app::MenuItem,
    utils::{self, Hwnd},
};

pub struct ContextMenu {
    hwnd: HWND,
    menu: HMENU,
    checkboxes: Vec<u32>,
    radio_groups: Vec<Vec<u32>>,
}

impl ContextMenu {
    pub fn new(hwnd: HWND, items: &[MenuItem]) -> Result<Self> {
        let (menu, checkboxes, radio_groups) = MenuBuilder::build(items)?;
        Ok(Self {
            hwnd,
            menu,
            checkboxes,
            radio_groups,
        })
    }

    pub fn show(&self) -> Option<(u32, bool)> {
        let (x, y) = utils::cursor_pos();
        let id = self.menu.popup(
            TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RETURNCMD,
            x,
            y,
            self.hwnd,
        );

        if let Some(id) = id {
            let state = self.update_state(id);
            Some((id, state))
        } else {
            None
        }
    }

    fn update_state(&self, id: u32) -> bool {
        if self.is_checkbox(id) {
            let check = !self.menu.checked(id);
            self.menu.check_item(id, check);
            check
        } else if let Some((first, last)) = self.find_radio_group(id) {
            self.menu.check_radio_item(first, last, id);
            true
        } else {
            false
        }
    }

    fn is_checkbox(&self, id: u32) -> bool {
        self.checkboxes.contains(&id)
    }

    fn find_radio_group(&self, id: u32) -> Option<(u32, u32)> {
        self.radio_groups
            .iter()
            .find(|&group| group.contains(&id))
            .map(|group| (*group.first().unwrap(), *group.last().unwrap()))
    }
}

struct MenuBuilder {
    menu: HMENU,
    checkboxes: Vec<u32>,
    radio_groups: Vec<Vec<u32>>,
}

impl MenuBuilder {
    fn build(items: &[MenuItem]) -> Result<(HMENU, Vec<u32>, Vec<Vec<u32>>)> {
        let mut builder = Self {
            menu: HMENU::new_popup()?,
            checkboxes: Vec::new(),
            radio_groups: Vec::new(),
        };

        for item in items {
            builder.build_item(item, builder.menu)?;
        }

        Ok((builder.menu, builder.checkboxes, builder.radio_groups))
    }

    fn build_item(&mut self, item: &MenuItem, menu: HMENU) -> Result<()> {
        match item {
            MenuItem::Action { id, text } => menu.append_action(*id, *text)?,
            MenuItem::CheckBox { id, text, checked } => {
                menu.append_checkbox(*id, *text, *checked)?;
                self.checkboxes.push(*id);
            }
            MenuItem::Radio { id, text, checked } => {
                menu.append_radio(*id, *text, *checked)?;

                if let Some(group) = self.radio_groups.last_mut() {
                    group.push(*id);
                } else {
                    self.radio_groups.push(vec![*id]);
                }
            }
            MenuItem::Separator => {
                menu.append_separator()?;
                self.add_new_radio_group();
            }
            MenuItem::SubMenu { text, items } => {
                let submenu = HMENU::new_popup()?;
                menu.append_submenu(*text, submenu)?;

                self.add_new_radio_group();

                for item in items {
                    self.build_item(item, submenu)?;
                }
            }
        }

        Ok(())
    }

    fn add_new_radio_group(&mut self) {
        let new_group = if let Some(group) = self.radio_groups.last() {
            !group.is_empty()
        } else {
            true
        };

        if new_group {
            self.radio_groups.push(vec![]);
        }
    }
}

pub trait Menu: Into<HMENU> {
    fn new_popup() -> Result<HMENU> {
        unsafe { CreatePopupMenu().map_err(anyhow::Error::msg) }
    }

    fn append_item(
        self,
        ftype: Option<MENU_ITEM_TYPE>,
        state: Option<MENU_ITEM_STATE>,
        id: Option<u32>,
        submenu: Option<HMENU>,
        data: Option<usize>,
        text: Option<PCSTR>,
    ) -> Result<()> {
        unsafe {
            let mi = MENUITEMINFOA::new(ftype, state, id, submenu, data, text);
            InsertMenuItemA(self.into(), u32::MAX, true, &mi as *const _)
                .map_err(anyhow::Error::msg)
        }
    }

    fn append_action(self, id: u32, text: PCSTR) -> Result<()> {
        self.append_item(Some(MFT_STRING), None, Some(id), None, None, Some(text))
    }

    fn append_checkbox(self, id: u32, text: PCSTR, checked: bool) -> Result<()> {
        self.append_item(
            Some(MFT_STRING),
            Some(if checked { MFS_CHECKED } else { MFS_UNCHECKED }),
            Some(id),
            None,
            None,
            Some(text),
        )
    }

    fn append_radio(self, id: u32, text: PCSTR, checked: bool) -> Result<()> {
        self.append_item(
            Some(MFT_STRING | MFT_RADIOCHECK),
            Some(if checked { MFS_CHECKED } else { MFS_UNCHECKED }),
            Some(id),
            None,
            None,
            Some(text),
        )
    }

    fn append_separator(self) -> Result<()> {
        self.append_item(Some(MFT_SEPARATOR), None, None, None, None, None)
    }

    fn append_submenu(self, text: PCSTR, submenu: HMENU) -> Result<()> {
        self.append_item(
            Some(MFT_STRING),
            None,
            None,
            Some(submenu),
            None,
            Some(text),
        )
    }

    fn popup(self, flags: TRACK_POPUP_MENU_FLAGS, x: i32, y: i32, hwnd: HWND) -> Option<u32> {
        match unsafe { TrackPopupMenu(self.into(), flags, x, y, None, hwnd, None) } {
            BOOL(0) => None,
            BOOL(id) => Some(id as u32),
        }
    }

    fn item_state(self, id: u32, state: MENU_ITEM_STATE) -> Result<MENUITEMINFOA> {
        unsafe {
            let mut mi = MENUITEMINFOA::new(None, Some(state), Some(id), None, None, None);
            GetMenuItemInfoA(self.into(), id, false, &mut mi)?;
            Ok(mi)
        }
    }

    fn set_item_state(self, id: u32, state: MENU_ITEM_STATE) -> Result<()> {
        unsafe {
            let mi = MENUITEMINFOA::new(None, Some(state), Some(id), None, None, None);
            SetMenuItemInfoA(self.into(), id, false, &mi).map_err(anyhow::Error::msg)
        }
    }

    fn checked(self, id: u32) -> bool {
        let mi = self.item_state(id, MFS_CHECKED).unwrap();
        (mi.fState & MFS_CHECKED) != MENU_ITEM_STATE(0)
    }

    fn check_item(self, id: u32, check: bool) {
        self.set_item_state(id, if check { MFS_CHECKED } else { MFS_UNCHECKED })
            .unwrap();
    }

    fn check_radio_item(self, first: u32, last: u32, id: u32) {
        unsafe {
            CheckMenuRadioItem(self.into(), first, last, id, MF_BYCOMMAND.0);
        }
    }
}

impl Menu for HMENU {}

pub trait MenuItemInfo {
    fn new(
        ftype: Option<MENU_ITEM_TYPE>,
        state: Option<MENU_ITEM_STATE>,
        id: Option<u32>,
        submenu: Option<HMENU>,
        data: Option<usize>,
        text: Option<PCSTR>,
    ) -> Self;
}

impl MenuItemInfo for MENUITEMINFOA {
    fn new(
        ftype: Option<MENU_ITEM_TYPE>,
        state: Option<MENU_ITEM_STATE>,
        id: Option<u32>,
        submenu: Option<HMENU>,
        data: Option<usize>,
        text: Option<PCSTR>,
    ) -> Self {
        let mask = {
            let mut mask = MENU_ITEM_MASK(0);
            if data.is_some() {
                mask |= MIIM_DATA;
            }
            if ftype.is_some() {
                mask |= MIIM_FTYPE;
            }
            if id.is_some() {
                mask |= MIIM_ID;
            }
            if state.is_some() {
                mask |= MIIM_STATE;
            }
            if text.is_some() {
                mask |= MIIM_STRING;
            }
            if submenu.is_some() {
                mask |= MIIM_SUBMENU;
            }
            mask
        };

        MENUITEMINFOA {
            cbSize: size_of::<MENUITEMINFOA>() as u32,
            fMask: mask,
            fType: ftype.unwrap_or_default(),
            fState: state.unwrap_or_default(),
            wID: id.unwrap_or_default(),
            hSubMenu: submenu.unwrap_or_default(),
            dwItemData: data.unwrap_or_default(),
            dwTypeData: text.map(|str| PSTR(str.0 as _)).unwrap_or(PSTR::null()),
            ..Default::default()
        }
    }
}
