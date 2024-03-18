use std::mem::size_of;

use anyhow::{Error as E, Result};
use windows::{
    core::PSTR,
    Win32::{Foundation::TRUE, UI::WindowsAndMessaging::*},
};

pub struct MenuBuilder<'a> {
    menu: HMENU,
    submenu: HMENU,
    checkboxes: &'a mut Vec<u32>,
    radio_groups: &'a mut Vec<Vec<u32>>,
}

impl<'a> MenuBuilder<'a> {
    pub fn new(
        menu: HMENU,
        checkboxes: &'a mut Vec<u32>,
        radio_groups: &'a mut Vec<Vec<u32>>,
    ) -> Self {
        Self {
            menu,
            submenu: HMENU(0),
            checkboxes,
            radio_groups,
        }
    }

    pub fn add_cmd<T: Into<u32>>(&mut self, id: T, text: &str) -> Result<&mut Self> {
        let id: u32 = id.into();
        let text = text.to_owned() + "\0";

        self.insert_menu_item(MenuItem::Cmd { id, text })?;

        Ok(self)
    }

    pub fn add_checkbox<T: Into<u32>>(
        &mut self,
        id: T,
        text: &str,
        checked: bool,
    ) -> Result<&mut Self> {
        let id: u32 = id.into();
        let text = text.to_owned() + "\0";

        self.insert_menu_item(MenuItem::Check { id, text, checked })?;
        self.checkboxes.push(id);

        Ok(self)
    }

    pub fn add_radio_group(&mut self) -> Result<&mut Self> {
        self.radio_groups.push(vec![]);
        Ok(self)
    }

    pub fn add_radio<T: Into<u32>>(
        &mut self,
        id: T,
        text: &str,
        checked: bool,
    ) -> Result<&mut Self> {
        let id: u32 = id.into();
        let text = text.to_owned() + "\0";

        self.insert_menu_item(MenuItem::Radio { id, text, checked })?;

        self.radio_groups.last_mut().unwrap().push(id);

        Ok(self)
    }

    pub fn push_submenu(&mut self, text: &str) -> Result<&mut Self> {
        if !self.submenu.is_invalid() {
            anyhow::bail!("multi level submenu is not supportted.");
        }

        let text = text.to_owned() + "\0";
        let submenu = unsafe { CreatePopupMenu() }?;

        self.insert_menu_item(MenuItem::SubMenu { submenu, text })?;
        self.submenu = submenu;

        Ok(self)
    }

    pub fn pop_submenu(&mut self) -> Result<&mut Self> {
        if self.submenu.is_invalid() {
            anyhow::bail!("no valid submenu.");
        }

        self.submenu = HMENU(0);
        Ok(self)
    }

    pub fn separate(&mut self) -> Result<&mut Self> {
        self.insert_menu_item(MenuItem::Separator)?;
        Ok(self)
    }

    fn insert_menu_item(&mut self, item: MenuItem) -> Result<()> {
        let menu = if self.submenu.is_invalid() {
            self.menu
        } else {
            self.submenu
        };

        insert_menu_item(menu, item)
    }
}

enum MenuItem {
    Cmd {
        id: u32,
        text: String,
    },
    Check {
        id: u32,
        text: String,
        checked: bool,
    },
    Radio {
        id: u32,
        text: String,
        checked: bool,
    },
    SubMenu {
        submenu: HMENU,
        text: String,
    },
    Separator,
}

fn insert_menu_item(menu: HMENU, mut item: MenuItem) -> Result<()> {
    unsafe {
        let (fmask, ftype, fstate, id, submenu, data) = match &mut item {
            MenuItem::Cmd { id, text } => (
                MIIM_TYPE | MIIM_ID,
                MFT_STRING,
                MFS_ENABLED,
                *id,
                HMENU(0),
                PSTR(text.as_mut_ptr()),
            ),
            MenuItem::Check { id, text, checked } => (
                MIIM_TYPE | MIIM_ID | MIIM_STATE,
                MFT_STRING,
                if *checked { MFS_CHECKED } else { MFS_ENABLED },
                *id,
                HMENU(0),
                PSTR(text.as_mut_ptr()),
            ),
            MenuItem::Radio { id, text, checked } => (
                MIIM_TYPE | MIIM_ID | MIIM_STATE,
                MFT_STRING | MFT_RADIOCHECK,
                if *checked { MFS_CHECKED } else { MFS_ENABLED },
                *id,
                HMENU(0),
                PSTR(text.as_mut_ptr()),
            ),
            MenuItem::SubMenu { submenu, text } => (
                MIIM_TYPE | MIIM_SUBMENU,
                MFT_STRING,
                MFS_ENABLED,
                0,
                *submenu,
                PSTR(text.as_mut_ptr()),
            ),
            MenuItem::Separator => (
                MIIM_FTYPE,
                MFT_SEPARATOR,
                MFS_ENABLED,
                0,
                HMENU(0),
                PSTR::null(),
            ),
        };

        let mi = MENUITEMINFOA {
            cbSize: size_of::<MENUITEMINFOA>() as u32,
            fMask: fmask,
            fType: ftype,
            fState: fstate,
            wID: id,
            hSubMenu: submenu,
            dwTypeData: data,
            ..Default::default()
        };

        InsertMenuItemA(menu, u32::MAX, TRUE, &mi).map_err(E::msg)
    }
}
