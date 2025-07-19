use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead};

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

// A struct to hold word vectors in a contiguous array for performance.
pub struct WordVectors {
    pub words: Vec<String>,           // vocabulary - index to word map
    word_map: HashMap<String, usize>, // word to index map
    pub vectors: Vec<f64>,            // A single, flattened Vec of all vector data
    pub dims: usize,                  // The dimension of each vector
}

impl WordVectors {
    pub fn get_index(&self, word: &str) -> Option<&usize> {
        self.word_map.get(word)
    }

    pub fn get_vector(&self, idx: usize) -> &[f64] {
        &self.vectors[idx * self.dims..(idx + 1) * self.dims]
    }

    pub fn from_file(filename: &str) -> Result<WordVectors, Box<dyn Error>> {
        const EPS: f64 = 1e-8;
        let file = fs::File::open(filename)?;
        let reader = io::BufReader::new(file);

        let mut words: Vec<String> = Vec::new();
        let mut word_map: HashMap<String, usize> = HashMap::new();
        let mut vectors_data: Vec<f64> = Vec::new(); // Accumulate all vector values here
        let mut dims: usize = 0; // Dimension will be determined from the first vector

        for (index, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let mut parts = line.trim().split_whitespace();

            if let Some(key) = parts.next() {
                let current_word = key.to_string();
                let mut values: Vec<f64> = parts
                    .map(|s| s.parse::<f64>())
                    .collect::<Result<Vec<f64>, _>>()?;

                if index == 0 {
                    // Determine dimensions from the first vector
                    dims = values.len();
                    if dims == 0 {
                        return Err("First vector has zero dimensions, cannot proceed.".into());
                    }
                } else {
                    // Ensure all subsequent vectors have the same dimension
                    if values.len() != dims {
                        return Err(format!(
                        "Vector for '{}' has dimension {} which differs from initial dimension {}",
                        current_word,
                        values.len(),
                        dims
                    )
                    .into());
                    }
                }

                let norm: f64 = values.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > EPS {
                    values.iter_mut().for_each(|e| *e /= norm);
                }

                // Store the word and its index
                word_map.insert(current_word.clone(), words.len());
                words.push(current_word);

                // Extend the flattened vector data
                vectors_data.extend_from_slice(&values);
            }
        }

        if words.is_empty() {
            return Err("No word vectors found in the file.".into());
        }

        Ok(WordVectors {
            words,
            word_map,
            vectors: vectors_data,
            dims,
        })
    }
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
