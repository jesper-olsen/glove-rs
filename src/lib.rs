use bytemuck::pod_read_unaligned;
use bytemuck::{Pod, Zeroable};
use byteorder::{LittleEndian, WriteBytesExt};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::{self, BufRead, ErrorKind, Read, Write};
use std::mem;

/// Co-occurrence record struct. `repr(C)` and `Pod` ensure the memory layout
/// is identical to the C struct, allowing us to read the binary file directly.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct Crec {
    pub word1: u32,
    pub word2: u32,
    pub val: f64,
}

impl Ord for Crec {
    fn cmp(&self, other: &Self) -> Ordering {
        // sort order: word 1 then word 2
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

impl Crec {
    /// Distinguishes between clean EOF and other I/O errors.
    /// - `Ok(Some(crec))`: Success.
    /// - `Ok(None)`: Clean End-Of-File.
    /// - `Err(e)`: An I/O error occurred.
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        let mut buffer = [0u8; mem::size_of::<Crec>()];
        match reader.read_exact(&mut buffer) {
            Ok(()) => {
                let crec = pod_read_unaligned(&buffer);
                Ok(Some(crec))
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }

    // write one CREC to a binary stream (File or Stdout)
    pub fn write<W: Write>(writer: &mut W, crec: &Crec) -> io::Result<()> {
        writer.write_u32::<LittleEndian>(crec.word1)?;
        writer.write_u32::<LittleEndian>(crec.word2)?;
        writer.write_f64::<LittleEndian>(crec.val)?;
        Ok(())
    }

    /// Writes a slice of Crec records to a writer in a single, efficient operation.
    pub fn write_slice<W: Write>(writer: &mut W, crecs: &[Crec]) -> io::Result<()> {
        // `unsafe` because we are asserting that the memory layout of a slice
        // of `Crec`s is equivalent to a contiguous slice of bytes.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(crecs.as_ptr() as *const u8, std::mem::size_of_val(crecs))
        };
        writer.write_all(byte_slice)
    }
}

// A struct to hold word vectors in a contiguous array for performance.
pub struct WordVectors {
    words: Vec<String>,               // vocabulary - index to word map
    word_map: HashMap<String, usize>, // word to index map
    vectors: Vec<f64>,                // A single, flattened Vec of all vector data
    dims: usize,                      // The dimension of each vector
}

impl WordVectors {
    pub fn get_word(&self, idx: usize) -> &str {
        &self.words[idx]
    }

    pub fn get_index(&self, word: &str) -> Option<&usize> {
        self.word_map.get(word)
    }

    fn get_vector(&self, idx: usize) -> &[f64] {
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
            let mut parts = line.split_whitespace();

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

    pub fn analogy(&self, a: &str, b: &str, c: &str) -> Option<usize> {
        // Get indices for our words. Compute analogy if none are OOV.
        let (Some(a_idx), Some(b_idx), Some(c_idx)) =
            (self.get_index(a), self.get_index(b), self.get_index(c))
        else {
            return None;
        };

        let va = self.get_vector(*a_idx);
        let vb = self.get_vector(*b_idx);
        let vc = self.get_vector(*c_idx);

        let mut target_vector = vec![0.0; self.dims];
        for i in 0..self.dims {
            target_vector[i] = vb[i] - va[i] + vc[i];
        }

        // Parallel search over contiguous memory ---
        let (_best_score, best_idx) = self
            .vectors
            .par_chunks_exact(self.dims) // Iterate over the vectors in parallel
            .enumerate() // Get the index of each vector
            .filter(|(i, _)| i != a_idx && i != b_idx && i != c_idx) // Filter out input words
            .map(|(i, v_slice)| {
                // Calculate dot product between the current word's vector and the target
                let score = v_slice
                    .iter()
                    .zip(&target_vector)
                    .map(|(v, t)| v * t)
                    .sum::<f64>();
                (score, i)
            })
            .reduce(
                || (-f64::INFINITY, 0), // (score, index)
                |best, current| if best.0 > current.0 { best } else { current },
            );

        Some(best_idx)
    }

    /// same as analogy, except it returns top n matches
    pub fn analogy_topn(&self, a: &str, b: &str, c: &str, n: usize) -> Option<Vec<(usize, f64)>> {
        // Get indices for our words. Compute analogy if none are OOV.
        let (Some(a_idx), Some(b_idx), Some(c_idx)) =
            (self.get_index(a), self.get_index(b), self.get_index(c))
        else {
            return None;
        };

        let va = self.get_vector(*a_idx);
        let vb = self.get_vector(*b_idx);
        let vc = self.get_vector(*c_idx);

        let mut target_vector = vec![0.0; self.dims];
        for i in 0..self.dims {
            target_vector[i] = vb[i] - va[i] + vc[i];
        }

        // Collect all scores in parallel
        let mut scores: Vec<(usize, f64)> = self
            .vectors
            .par_chunks_exact(self.dims)
            .enumerate()
            .filter(|(i, _)| i != a_idx && i != b_idx && i != c_idx)
            .map(|(i, v_slice)| {
                let score = v_slice
                    .iter()
                    .zip(&target_vector)
                    .map(|(v, t)| v * t)
                    .sum::<f64>();
                (i, score)
            })
            .collect();

        // Sort by score in descending order
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top N results
        scores.truncate(n);

        Some(scores)
    }

    pub fn nearest_to_sum(&self, words: &[&str], n: usize) -> Option<Vec<(usize, f64)>> {
        // Get indices for all words, collecting valid ones
        let indices: Vec<usize> = words
            .iter()
            .filter_map(|word| self.get_index(word).copied())
            .collect();

        // Return None if no valid words found
        if indices.is_empty() {
            return None;
        }

        // Sum all the word vectors
        let mut target_vector = vec![0.0; self.dims];
        for &idx in &indices {
            let word_vec = self.get_vector(idx);
            for i in 0..self.dims {
                target_vector[i] += word_vec[i];
            }
        }

        // Normalize the target vector for cosine similarity
        let magnitude: f64 = target_vector.iter().map(|x| x * x).sum::<f64>().sqrt();

        if magnitude == 0.0 {
            return None;
        }

        for e in target_vector.iter_mut().take(self.dims) {
            *e /= magnitude
        }

        // Collect all scores in parallel
        let mut scores: Vec<(usize, f64)> = self
            .vectors
            .par_chunks_exact(self.dims)
            .enumerate()
            .filter(|(i, _)| !indices.contains(i)) // Exclude all input words
            .map(|(i, v_slice)| {
                // Compute cosine similarity (assuming vectors are normalized)
                let score = v_slice
                    .iter()
                    .zip(&target_vector)
                    .map(|(v, t)| v * t)
                    .sum::<f64>();
                (i, score)
            })
            .collect();

        // Sort by score in descending order
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top N results
        scores.truncate(n);

        Some(scores)
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
