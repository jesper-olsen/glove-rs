use glove_rs::WordVectors;
use std::fs;
use std::io::{self, BufRead};

// Read 4-word analogy test file like the ones used in the google analogy corpus:
// Efficient Estimation of Word Representations in Vector Space
// Tomas Mikolov et al, 2013
// e.g. "King is to Queen as Man is to Woman"
// one test per line
fn read_analogy_test(fname: &str) -> io::Result<Vec<(String, String, String, String)>> {
    let file = fs::File::open(fname)?;
    let reader = io::BufReader::new(file);
    let mut r = Vec::new();

    for line in reader.lines() {
        let line = line?;
        // Skip section headers like ": capital-common-countries"
        if line.starts_with(':') {
            continue;
        }

        let mut parts = line.split_whitespace().map(str::to_owned);
        if let (Some(a), Some(b), Some(c), Some(d)) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        {
            r.push((a, b, c, d));
        }
    }

    Ok(r)
}

const SEMANTIC_TESTS: [&str; 5] = [
    "capital-common-countries.txt",
    "capital-world.txt",
    "currency.txt",
    "city-in-state.txt",
    "family.txt",
];

const SYNTACTIC_TESTS: [&str; 9] = [
    "gram1-adjective-to-adverb.txt",
    "gram2-opposite.txt",
    "gram3-comparative.txt",
    "gram4-superlative.txt",
    "gram5-present-participle.txt",
    "gram6-nationality-adjective.txt",
    "gram7-past-tense.txt",
    "gram8-plural.txt",
    "gram9-plural-verbs.txt",
];

/// Test model on all categories in the google analogy corpus
fn analogy_test(models: &WordVectors) -> Result<(), std::io::Error> {
    fn run_test_group(
        label: &str,
        files: &[&str],
        models: &WordVectors,
    ) -> Result<(usize, usize, usize), std::io::Error> {
        println!("\n{label} ANALOGY TESTS");
        let mut group_correct = 0;
        let mut group_total = 0;
        let mut group_seen = 0;

        for fname in files {
            let path = format!("DATA/question-data/{fname}");
            let (correct, seen, total) = analogy_test_file(models, &path)?;
            println!(
                "File: {fname} — Accuracy: {:4.2}% ({correct}/{seen})",
                100.0 * correct as f64 / seen as f64
            );
            group_correct += correct;
            group_seen += seen;
            group_total += total;
        }

        println!(
            "{label} Total Accuracy: {:4.2}% ({group_correct}/{group_seen})",
            100.0 * group_correct as f64 / group_seen as f64
        );
        println!(
            "{label} Questions seen/total: {:4.2}% ({group_seen}/{group_seen})",
            100.0 * group_seen as f64 / group_total as f64
        );

        Ok((group_correct, group_seen, group_total))
    }

    let (sem_correct, sem_seen, sem_total) = run_test_group("SEMANTIC", &SEMANTIC_TESTS, models)?;
    let (syn_correct, syn_seen, syn_total) = run_test_group("SYNTACTIC", &SYNTACTIC_TESTS, models)?;

    let total_correct = sem_correct + syn_correct;
    let total_seen = sem_seen + syn_seen;
    let total_total = sem_total + syn_total;

    println!("\nOVERALL RESULTS:");
    println!(
        "Total Accuracy: {:4.2}% ({total_correct}/{total_seen})",
        100.0 * total_correct as f64 / total_seen as f64
    );
    println!(
        "Total Questions seen/total: {:4.2}% ({total_seen}/{total_total})",
        100.0 * total_seen as f64 / total_total as f64
    );

    Ok(())
}

fn analogy_test_file(
    word_vectors: &WordVectors,
    fname: &str,
) -> Result<(usize, usize, usize), std::io::Error> {
    let mut correct = 0;
    let mut total = 0;
    let mut seen = 0;

    for (a, b, c, d) in read_analogy_test(fname)? {
        total += 1;

        let Some(&answer_idx) = word_vectors.get_index(&d) else {
            continue;
        };
        let Some(guess_idx) = word_vectors.analogy(&a, &b, &c) else {
            continue;
        };
        if guess_idx == answer_idx {
            correct += 1
        }
        seen += 1;
    }
    Ok((correct, seen, total))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let word_vectors = WordVectors::from_file("vectors.txt")?;
    analogy_test(&word_vectors)?;
    Ok(())
}
