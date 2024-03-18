use windows::Win32::{Foundation::*, UI::WindowsAndMessaging::CREATESTRUCTA};

#[allow(unused, non_snake_case)]
pub fn MAKELONG(lo: u16, hi: u16) -> usize {
    (lo as usize & 0xffff) | ((hi as usize & 0xffff) << 16)
}

#[allow(unused, non_snake_case)]
pub fn LOWORD(dw: usize) -> i32 {
    (dw & 0xffff) as _
}

#[allow(unused, non_snake_case)]
pub fn HIWORD(dw: usize) -> i32 {
    ((dw >> 16) & 0xffff) as _
}

#[allow(unused, non_snake_case)]
pub fn MAKEWPARAM(lo: u16, hi: u16) -> WPARAM {
    WPARAM(MAKELONG(lo, hi))
}

#[allow(unused, non_snake_case)]
pub fn BREAKWP(wp: WPARAM) -> (i32, i32) {
    (LOWORD(wp.0 as _), HIWORD(wp.0 as _))
}

#[allow(unused, non_snake_case)]
pub fn MAKELPARAM(lo: u16, hi: u16) -> LPARAM {
    LPARAM(MAKELONG(lo, hi) as _)
}

#[allow(unused, non_snake_case)]
pub fn BREAKLPARAM(lp: LPARAM) -> (i32, i32) {
    (LOWORD(lp.0 as _), HIWORD(lp.0 as _))
}

#[allow(unused, non_snake_case)]
pub fn GET_X_LPARAM(lp: LPARAM) -> i32 {
    LOWORD(lp.0 as _) as _
}

#[allow(unused, non_snake_case)]
pub fn GET_Y_LPARAM(lp: LPARAM) -> i32 {
    HIWORD(lp.0 as _) as _
}

#[allow(unused, non_snake_case)]
pub fn rect_width(rect: &RECT) -> i32 {
    rect.right - rect.left
}

#[allow(unused, non_snake_case)]
pub fn rect_height(rect: &RECT) -> i32 {
    rect.bottom - rect.top
}

#[allow(unused, non_snake_case)]
pub fn rect_size(rect: &RECT) -> (i32, i32) {
    (rect_width(rect), rect_height(rect))
}

#[allow(unused, non_snake_case)]
pub fn wheel_delta(wp: WPARAM) -> i32 {
    ((wp.0 >> 16) & 0xffff) as i16 as i32
}

#[allow(unused, non_snake_case)]
pub fn create_struct(lp: LPARAM) -> *const CREATESTRUCTA {
    unsafe { lp.0 as *const CREATESTRUCTA }
}

#[allow(unused, non_snake_case)]
pub fn create_param<T>(lp: LPARAM) -> *mut T {
    unsafe { std::mem::transmute((*create_struct(lp)).lpCreateParams) }
}
