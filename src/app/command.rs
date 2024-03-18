use super::config::*;

#[repr(u32)]
#[derive(Copy, Clone, PartialEq, PartialOrd)]
pub enum Command {
    Noop,
    Clear,
    Quit,
    ModelDistilSmallEn,
    ModelDistilMediumEn,
    ModelDistilLargeV2,
    ModelWhisperLargeV3,
    DelayNone,
    DelayLow,
    DelayMid,
    DelayHigh,
    DelayHighest,
    TransparencyNone,
    TransparencyLow,
    TransparencyMid,
    TransparencyHigh,
    Transparent,
    FontFamilyAptos,
    FontFamilyArial,
    FontFamilyCalibri,
    FontFamilySegoeUI,
    FontFamilyCambria,
    FontFamilyGeorgia,
    FontFamilyConsola,
    FontFamilyComicSansMS,
    FontFamilyImpact,
    FontSizeVerySmall,
    FontSizeSmall,
    FontSizeMedium,
    FontSizeLarge,
    FontSizeVeryLarge,
    TextStyleBold,
    TextStyleItalic,
    TextStyleOutline,
    Max,
}

use Command::*;

impl From<Command> for Model {
    fn from(value: Command) -> Self {
        match value {
            ModelDistilSmallEn => Model::DistilSmallEn,
            ModelDistilMediumEn => Model::DistilMediumEn,
            ModelDistilLargeV2 => Model::DistilLargeV2,
            ModelWhisperLargeV3 => Model::WhisperLargeV3,
            _ => Default::default(),
        }
    }
}

impl From<Command> for Delay {
    fn from(value: Command) -> Self {
        match value {
            DelayNone => Delay::None,
            DelayLow => Delay::Low,
            DelayMid => Delay::Mid,
            DelayHigh => Delay::High,
            DelayHighest => Delay::Highest,
            _ => Default::default(),
        }
    }
}

impl From<Command> for Transparency {
    fn from(value: Command) -> Self {
        match value {
            TransparencyNone => Transparency::None,
            TransparencyLow => Transparency::Low,
            TransparencyMid => Transparency::Mid,
            TransparencyHigh => Transparency::High,
            Transparent => Transparency::Transparent,
            _ => Default::default(),
        }
    }
}

impl From<Command> for FontFamily {
    fn from(value: Command) -> Self {
        match value {
            FontFamilyAptos => FontFamily::Aptos,
            FontFamilyArial => FontFamily::Arial,
            FontFamilyCalibri => FontFamily::Calibri,
            FontFamilySegoeUI => FontFamily::SegoeUI,
            FontFamilyCambria => FontFamily::Cambria,
            FontFamilyGeorgia => FontFamily::Georgia,
            FontFamilyConsola => FontFamily::Consola,
            FontFamilyComicSansMS => FontFamily::ComicSansMS,
            FontFamilyImpact => FontFamily::Impact,
            _ => Default::default(),
        }
    }
}

impl From<Command> for FontSize {
    fn from(value: Command) -> Self {
        match value {
            FontSizeVerySmall => FontSize::VerySmall,
            FontSizeSmall => FontSize::Small,
            FontSizeMedium => FontSize::Medium,
            FontSizeLarge => FontSize::Large,
            FontSizeVeryLarge => FontSize::VeryLarge,
            _ => Default::default(),
        }
    }
}

impl From<Command> for u32 {
    fn from(value: Command) -> Self {
        value as _
    }
}

impl From<u32> for Command {
    fn from(id: u32) -> Self {
        if (Command::Noop as u32) < id && id < (Command::Max as u32) {
            unsafe { std::mem::transmute(id) }
        } else {
            Command::Noop
        }
    }
}
