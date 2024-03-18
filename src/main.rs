#![windows_subsystem = "windows"]

use anyhow::Result;
use windows::Win32::System::Com::*;

fn main() -> Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE).ok()?;
    }

    livesub::gui::window::run_app::<livesub::app::App>()
}
