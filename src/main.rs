// #![windows_subsystem = "windows"]

mod asr;
mod audio;
mod config;
mod controller;
mod renderer;
mod window;

use anyhow::Result;
use windows::{
    Win32::{
        System::WinRT::{
            CreateDispatcherQueueController, DQTAT_COM_NONE, DQTYPE_THREAD_CURRENT,
            DispatcherQueueOptions, RO_INIT_SINGLETHREADED, RoInitialize, RoUninitialize,
        },
        UI::WindowsAndMessaging::{DispatchMessageA, GetMessageA, MSG, TranslateMessage},
    },
    core::BOOL,
};

use crate::{config::Config, window::Window};
fn win_main() -> Result<()> {
    unsafe {
        RoInitialize(RO_INIT_SINGLETHREADED)?;
        let controller = CreateDispatcherQueueController(DispatcherQueueOptions {
            dwSize: std::mem::size_of::<DispatcherQueueOptions>() as _,
            threadType: DQTYPE_THREAD_CURRENT,
            apartmentType: DQTAT_COM_NONE,
        })?;
        let _dispatcher_queue = controller.DispatcherQueue()?;

        let config = Box::new(Config::load()?);
        Window::create(config)?;

        loop {
            let mut msg = MSG::default();
            match GetMessageA(&mut msg, None, 0, 0) {
                BOOL(0) | BOOL(-1) => {
                    break;
                }
                _ => {
                    _ = TranslateMessage(&msg);
                    _ = DispatchMessageA(&msg);
                }
            }
        }

        RoUninitialize();
        Ok(())
    }
}

fn main() {
    if let Err(e) = win_main() {
        eprintln!("{e:?}");
    }
}
