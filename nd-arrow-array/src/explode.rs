#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExplodeArgs {
    pub repeat_elems: usize,
    pub repeat_slices: usize,
}
