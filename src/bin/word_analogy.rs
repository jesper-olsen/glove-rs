use glove_rs::WordVectors;
use std::io::{self, Write};

fn get_input() -> io::Result<String> {
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    Ok(s.trim().to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let word_vectors = WordVectors::from_file("vectors.txt")?;

    loop {
        println!("\nWord analogy - KING is to QUEEN as MAN is to ?");
        print!("Enter 3 words: ");
        io::stdout().flush().unwrap();
        let s = get_input()?;
        if s == "EXIT" {
            break;
        }
        let words: Vec<&str> = s.trim().split_whitespace().collect();
        if words.len() < 3 {
            println!("Only {} words were input. Try again", words.len());
            continue;
        }

        match words.len() {
            3 => (),
            n => {
                println!("Expected exactly 3 words, but got {}. Try again.", n);
                continue;
            }
        }

        let oov_words: Vec<&str> = words
            .iter()
            .take(3) // Only check first 3 words
            .filter(|&&w| word_vectors.get_index(w).is_none())
            .copied()
            .collect();

        if !oov_words.is_empty() {
            for word in &oov_words {
                println!("'{}' is out of vocabulary", word);
            }
            continue;
        }

        let Some(topn) = word_vectors.analogy_topn(&words[0], &words[1], &words[2], 30) else {
            println!("No analogies");
            continue;
        };

        for (i, (idx, score)) in topn.iter().enumerate() {
            println!("{:3}: {:>8.5} {}", i + 1, score, word_vectors.words[*idx]);
        }
    }

    Ok(())
}
