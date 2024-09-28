use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct TextStream(Arc<Mutex<TextStreamInner>>);

impl TextStream {
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(TextStreamInner::new())))
    }

    pub fn set(&mut self, text: String, is_new_segment: bool) {
        if let Ok(mut inner) = self.0.lock() {
            inner.set(text, is_new_segment);
        }
    }

    pub fn get(&mut self) -> Option<String> {
        if let Ok(mut inner) = self.0.lock() {
            inner.get()
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        if let Ok(mut inner) = self.0.lock() {
            inner.clear();
        }
    }
}

struct TextStreamInner {
    prev: String,
    cur: String,
    dirty: bool,
}

impl TextStreamInner {
    fn new() -> Self {
        Self {
            prev: String::new(),
            cur: String::new(),
            dirty: false,
        }
    }

    fn set(&mut self, text: String, is_new_segment: bool) {
        if is_new_segment {
            self.prev = self.cur.clone();
            self.cur.clear();
            self.dirty = true;
        }

        if self.cur != text {
            self.cur = text;
            self.dirty = true;
        }
    }

    fn get(&mut self) -> Option<String> {
        if self.dirty {
            self.dirty = false;
            Some(self.prev.clone() + &self.cur)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.prev.clear();
        self.cur.clear();
        self.dirty = true;
    }
}
