use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_start_token_id: usize,
    pub d_model: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub suppress_tokens: Option<Vec<u32>>,
    pub begin_suppress_tokens: Option<Vec<u32>>,
}

impl Config {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&config)?;
        Ok(config)
    }
}
