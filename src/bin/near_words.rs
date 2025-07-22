use glove_rs::WordVectors;
use std::io::{self, Write};

fn get_input() -> io::Result<String> {
    let mut s = String::new();
    io::stdin().read_line(&mut s)?;
    Ok(s.trim().to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let word_vectors = WordVectors::from_file("vectors.txt")?;

    println!("Near Words Tool - Type 'EXIT' to quit\n");
    loop {
        println!("\nRanking nearest words to a word or sentence.");
        print!("Enter 1 or more words: ");
        io::stdout().flush().unwrap();
        let s = get_input()?;
        if s == "EXIT" {
            println!("Goodbye!");
            break;
        }
        let words: Vec<&str> = s.split_whitespace().collect();
        if words.is_empty() {
            println!("No words were input. Try again");
            continue;
        }

        let oov_words: Vec<&str> = words
            .iter()
            .filter(|&&w| word_vectors.get_index(w).is_none())
            .copied()
            .collect();

        if !oov_words.is_empty() {
            for word in &oov_words {
                println!("'{word}' is out of vocabulary");
            }
            continue;
        }

        const TOP_N: usize = 30;
        let Some(topn) = word_vectors.nearest_to_sum(&words, TOP_N) else {
            println!("No near words!");
            continue;
        };

        println!("\nNearest words to '{}':", words.join(" + "));
        println!("{:>4} {:>10} Word", "Rank", "Score");
        println!("{}", "-".repeat(30));

        for (i, (idx, score)) in topn.iter().enumerate() {
            println!(
                "{:4}: {:10.6} {}",
                i + 1,
                score,
                word_vectors.get_word(*idx)
            );
        }
    }

    Ok(())
}
