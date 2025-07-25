use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

#[derive(Parser, Debug)]
#[command(author, version, about = "Extract vocabulary from text corpus", long_about = None)]
struct Args {
    /// Minimum count threshold for words to be included in vocabulary
    #[arg(short, long, default_value_t = 1)]
    min_count: usize,

    /// Input file path (if not provided, reads from stdin)
    #[arg(value_name = "FILE")]
    input: Option<String>,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    // Create reader based on input source
    let reader: Box<dyn BufRead> = match args.input {
        Some(filename) => {
            let file = File::open(filename)?;
            Box::new(BufReader::new(file))
        }
        None => Box::new(BufReader::new(io::stdin())),
    };

    let word_counts = count_words(reader)?;

    let mut vocabulary: Vec<(String, usize)> = word_counts
        .into_iter()
        .filter(|(_, count)| *count >= args.min_count)
        .collect();

    // Sort by frequency (descending) and then alphabetically
    vocabulary.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    for (word, count) in vocabulary {
        println!("{word}\t{count}");
    }

    Ok(())
}

fn count_words(reader: Box<dyn BufRead>) -> io::Result<HashMap<String, usize>> {
    let mut word_counts = HashMap::new();

    let mut content = String::new();
    let mut reader = reader;
    reader.read_to_string(&mut content)?;

    for word in content.split_whitespace() {
        let word = word.to_lowercase();

        let word = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>();

        if !word.is_empty() {
            *word_counts.entry(word).or_insert(0) += 1;
        }
    }

    Ok(word_counts)
}
