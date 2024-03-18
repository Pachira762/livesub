#[derive(Default)]
pub struct TextStream {
    segment: String,
    text: String,
    modified: bool,
}

impl TextStream {
    pub fn new() -> Self {
        TextStream {
            segment: String::new(),
            text: String::new(),
            modified: false,
        }
    }

    pub fn clear(&mut self) {
        self.segment.clear();
        self.text.clear();
        self.modified = true;
    }

    pub fn set(&mut self, new_text: String, new_segment: bool) {
        if new_segment {
            let new_segment = self.text.to_owned();
            let n_segment = new_segment.len() - count_overlapped(&new_segment, &new_text);

            self.segment = new_segment[..n_segment].to_owned();
        }

        if self.text != new_text {
            self.text = new_text;
            self.modified = true;
        }
    }

    pub fn set_force(&mut self, new_text: String) {
        self.segment.clear();
        self.text = new_text;
        self.modified = true;
    }

    pub fn get(&mut self) -> Option<String> {
        if self.modified {
            self.modified = false;
            let text = self.segment.to_owned() + &self.text;
            let text = replace_sentence_endings_with_newlines(&text);

            Some(text)
        } else {
            None
        }
    }
}

fn count_overlapped(a: &str, b: &str) -> usize {
    if let Some(word) = b.split_whitespace().next() {
        if let Some(i) = a.to_ascii_lowercase().rfind(&word.to_ascii_lowercase()) {
            return a.len() - i;
        }
    }

    0
}

fn replace_sentence_endings_with_newlines(text: &str) -> String {
    text.replace(". ", ".\r\n")
        .replace("! ", "!\r\n")
        .replace("? ", "?\r\n")
}
