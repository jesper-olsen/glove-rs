use std::cmp::Ordering;

// Deriving Ord allows for direct sorting. The order is important: word1 then word2.
#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub struct Crec {
    pub word1: i32,
    pub word2: i32,
    pub val: f64, // Using f64 for `real` type
}

// Custom Ord implementation to match C's `compare_crec`
impl Ord for Crec {
    fn cmp(&self, other: &Self) -> Ordering {
        self.word1
            .cmp(&other.word1)
            .then_with(|| self.word2.cmp(&other.word2))
    }
}

impl PartialOrd for Crec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Manual implementation of PartialEq.
// We ONLY compare word1 and word2, ignoring `val`.
impl PartialEq for Crec {
    fn eq(&self, other: &Self) -> bool {
        self.word1 == other.word1 && self.word2 == other.word2
    }
}

impl Eq for Crec {} // Marker trait - necessary for Ord

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
