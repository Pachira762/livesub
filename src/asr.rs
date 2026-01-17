mod common;
mod parakeet;
mod reazonspeech;
mod silero_vad;
pub mod transcribe;

pub fn shutdow() {
    self::common::cudnn_ctx::shutdown();
}
