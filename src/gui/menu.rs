use anyhow::Result;
use windows::{
    core::{s, PCSTR},
    Win32::{
        Foundation::{FreeLibrary, BOOL, HWND},
        System::LibraryLoader::{GetProcAddress, LoadLibraryExA, LOAD_LIBRARY_SEARCH_SYSTEM32},
        UI::WindowsAndMessaging::*,
    },
};

use super::MenuBuilder;

pub struct Menu {
    menu: HMENU,
    checkboxes: Vec<u32>,
    radio_groups: Vec<Vec<u32>>,
}

impl Menu {
    pub fn new() -> Result<Self> {
        unsafe {
            #[repr(C)]
            enum PreferredAppMode {
                #[allow(unused)]
                Default,
                #[allow(unused)]
                AllowDark,
                #[allow(unused)]
                ForceDark,
                #[allow(unused)]
                ForceLight,
                #[allow(unused)]
                Max,
            }
            type FnSetPreferredAppMode = extern "stdcall" fn(PreferredAppMode) -> PreferredAppMode;

            if let Ok(lib) = LoadLibraryExA(s!("uxtheme.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32) {
                if let Some(func) = GetProcAddress(lib, PCSTR(135 as *const u8)) {
                    let func = std::mem::transmute::<_, FnSetPreferredAppMode>(func);
                    func(PreferredAppMode::AllowDark);
                }

                FreeLibrary(lib)?;
            }

            let menu = CreatePopupMenu()?;

            Ok(Self {
                menu,
                checkboxes: vec![],
                radio_groups: vec![],
            })
        }
    }

    pub fn show<T: From<u32>>(&self, hwnd: HWND) -> Option<(T, Option<bool>)> {
        unsafe {
            let mut pos = Default::default();
            _ = GetCursorPos(&mut pos);

            match TrackPopupMenu(
                self.menu,
                TPM_LEFTALIGN | TPM_TOPALIGN | TPM_RETURNCMD,
                pos.x,
                pos.y,
                0,
                hwnd,
                None,
            ) {
                BOOL(0) => None,
                BOOL(id) => {
                    let state = self.on_command(id as u32);
                    Some((T::from(id as u32), state))
                }
            }
        }
    }

    pub fn get_builder(&mut self) -> MenuBuilder {
        MenuBuilder::new(self.menu, &mut self.checkboxes, &mut self.radio_groups)
    }

    fn on_command(&self, id: u32) -> Option<bool> {
        if let Some((first, last)) = self.find_radio_group(id) {
            _ = unsafe { CheckMenuRadioItem(self.menu, first, last, id, MF_BYCOMMAND.0) };
            Some(true)
        } else if self.find_checkbox(id) {
            let check = !self.is_checked(id);
            let ucheck = if check {
                MF_BYCOMMAND | MF_CHECKED
            } else {
                MF_BYCOMMAND
            };
            _ = unsafe { CheckMenuItem(self.menu, id, ucheck.0) };
            Some(check)
        } else {
            None
        }
    }

    fn find_checkbox(&self, id: u32) -> bool {
        self.checkboxes.contains(&id)
    }

    fn find_radio_group(&self, id: u32) -> Option<(u32, u32)> {
        for group in self.radio_groups.iter() {
            for item_id in group {
                if id == *item_id {
                    let first = *group.first().unwrap();
                    let last = *group.last().unwrap();
                    return Some((first, last));
                }
            }
        }
        None
    }

    fn is_checked(&self, id: u32) -> bool {
        let state = unsafe { GetMenuState(self.menu, id, MF_BYCOMMAND) };
        MENU_ITEM_FLAGS(state) & MF_CHECKED != MENU_ITEM_FLAGS(0)
    }
}
