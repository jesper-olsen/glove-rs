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
    word2index: HashMap<&'static str, usize>,
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

    fn leak_string(s: String) -> &'static str {
        Box::leak(Box::new(s))
    }

    pub fn add(&mut self, word: &str) -> usize {
        if let Some(&idx) = self.word2index.get(word) {
            return idx;
        }
        let idx = self.size();
        let word_owned = word.to_string();
        let word_ref = Self::leak_string(word_owned);
        self.word2index.insert(word_ref, idx);
        self.index2word.push(word_ref.to_string());
        idx
    }

    /// Returns the integer ID for a given word.
    #[inline]
    pub fn get_index(&self, word: &str) -> Option<usize> {
        self.word2index.get(word).copied()
    }

    /// Returns the word for a given integer ID.
    #[inline]
    pub fn get_word(&self, idx: usize) -> Option<&str> {
        self.index2word.get(idx).map(|s| s.as_str())
    }

    /// Returns the total number of words in the vocabulary.
    #[inline]
    pub fn size(&self) -> usize {
        self.index2word.len()
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
        assert_eq!(vocab.get_index("one"), Some(idx1));
        assert_eq!(vocab.get_index("two"), Some(idx2));
        assert_eq!(vocab.get_index("three"), Some(idx3));
        let s = vocab.get_word(idx3).unwrap();
        assert_eq!("three", s);
    }
}
