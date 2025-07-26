mod crec;
pub use crate::crec::Crec;
mod word_vectors;
pub use crate::word_vectors::WordVectors;

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Default)]
pub struct Vocabulary {
    word2index: HashMap<String, usize>,
    index2word: Vec<String>,
}

impl Vocabulary {
    /// Loads a vocabulary from a file, creating a mapping from words to 0-based integer indices.
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let mut vocab = Vocabulary::default();
        let vocab_file = File::open(path)?;
        for line in BufReader::new(vocab_file).lines() {
            if let Some(word) = line?.split_whitespace().next() {
                vocab.add(word);
            }
        }
        Ok(vocab)
    }

    pub fn add(&mut self, word: &str) -> usize {
        if let Some(&idx) = self.word2index.get(word) {
            return idx;
        }
        let idx = self.size();
        self.word2index.insert(word.to_string(), idx);
        self.index2word.push(word.to_string());
        idx
    }

    /// Returns the integer ID for a given word.
    #[inline]
    pub fn get_index(&self, word: &str) -> Option<&usize> {
        self.word2index.get(word)
    }

    /// Returns the integer ID for a given word.
    #[inline]
    pub fn get_word(&self, idx: usize) -> Option<&str> {
        if idx >= self.index2word.len() {
            return None;
        }
        Some(self.index2word[idx].as_str())
    }

    /// Returns the total number of words in the vocabulary.
    #[inline]
    pub fn size(&self) -> usize {
        self.word2index.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::Vocabulary;

    #[test]
    fn test_add_and_lookup() {
        let mut vocab = Vocabulary::default();
        let idx1 = vocab.add("one");
        let idx2 = vocab.add("two");
        let idx3 = vocab.add("three");
        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.get_index("one"), Some(&idx1));
        assert_eq!(vocab.get_index("two"), Some(&idx2));
        assert_eq!(vocab.get_index("three"), Some(&idx3));
    }
}
