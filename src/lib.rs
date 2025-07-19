use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

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

// read vectors and normalise to unit norms
pub fn read_vectors(filename: &str) -> Result<HashMap<String, Vec<f64>>, Box<dyn Error>> {
    const EPS: f64 = 1e-8;
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let mut vectors = HashMap::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let mut parts = line.trim().split_whitespace();

        if let Some(key) = parts.next() {
            let mut values = parts
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<f64>, _>>()?;
            let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > EPS {
                values.iter_mut().for_each(|e| *e /= norm);
            }
            vectors.insert(key.to_string(), values);
        }
    }

    Ok(vectors)
}

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
