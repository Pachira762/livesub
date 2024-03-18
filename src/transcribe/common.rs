pub trait Shiftable {
    fn shift(&mut self, n: usize);
}

impl<T: Copy> Shiftable for Vec<T> {
    fn shift(&mut self, n: usize) {
        let remain = self.len() - n;
        self.copy_within(n.., 0);
        self.truncate(remain);
    }
}
