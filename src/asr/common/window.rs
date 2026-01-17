use std::f32::consts::PI;

pub fn hanning(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (len - 1) as f32).cos())
        .collect()
}

pub fn povey(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f32 / (len - 1) as f32).cos()).powf(0.85))
        .collect()
}
