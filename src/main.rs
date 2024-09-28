#![windows_subsystem = "windows"]

pub mod app;
pub mod config;
pub mod graphics;
pub mod gui;
pub mod speech_to_text;

use anyhow::Result;

fn main() -> Result<()> {
    gui::run_app::<app::App>()
}
