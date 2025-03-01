pub mod app;
mod menu;
pub mod utils;
mod viewer;
mod window;

use anyhow::Result;
use viewer::Viewer;
use windows::Win32::{
    Foundation::FreeLibrary,
    System::{
        LibraryLoader::{GetProcAddress, LoadLibraryExA, LOAD_LIBRARY_SEARCH_SYSTEM32},
        WinRT::{
            CreateDispatcherQueueController, DispatcherQueueOptions, RoInitialize, DQTAT_COM_NONE,
            DQTYPE_THREAD_CURRENT, RO_INIT_MULTITHREADED,
        },
    },
    UI::WindowsAndMessaging::{DispatchMessageA, GetMessageA, TranslateMessage, MSG},
};
use windows_core::{s, BOOL, PCSTR};

use crate::config::Config;

pub fn run_app<T: app::App>() -> Result<()> {
    unsafe {
        RoInitialize(RO_INIT_MULTITHREADED)?;

        let controller = CreateDispatcherQueueController(DispatcherQueueOptions {
            dwSize: std::mem::size_of::<DispatcherQueueOptions>() as _,
            threadType: DQTYPE_THREAD_CURRENT,
            apartmentType: DQTAT_COM_NONE,
        })?;
        let _dispatcher_queu = controller.DispatcherQueue()?;

        set_preferred_app_mode(PreferredAppMode::AllowDark)?;

        let config = Config::load();
        let _viewer = Viewer::<T>::create(config)?;

        loop {
            let mut msg = MSG::default();
            match GetMessageA(&mut msg, None, 0, 0) {
                BOOL(0) | BOOL(-1) => break,
                _ => {
                    _ = TranslateMessage(&msg);
                    DispatchMessageA(&msg);
                }
            }
        }
    }

    Ok(())
}

#[allow(unused)]
#[repr(C)]
enum PreferredAppMode {
    Default,
    AllowDark,
    ForceDark,
    ForceLight,
    Max,
}

fn set_preferred_app_mode(mode: PreferredAppMode) -> Result<()> {
    type FnSetPreferredAppMode = extern "stdcall" fn(PreferredAppMode) -> PreferredAppMode;

    unsafe {
        if let Ok(lib) = LoadLibraryExA(s!("uxtheme.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32) {
            if let Some(addr) = GetProcAddress(lib, PCSTR(135 as *const u8)) {
                std::mem::transmute::<unsafe extern "system" fn() -> isize, FnSetPreferredAppMode>(
                    addr,
                )(mode);
            }
            FreeLibrary(lib)?;
        }
    }

    Ok(())
}
