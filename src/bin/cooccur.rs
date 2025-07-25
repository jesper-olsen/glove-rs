use clap::Parser;
use glove_rs::{Crec, Vocabulary};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::mem;

/// The clap struct for parsing command-line arguments.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Set verbosity level
    #[arg(short, long, default_value_t = 2)]
    verbose: u8,

    /// Use symmetric context window
    #[arg(long)]
    asymmetric: bool,

    /// Set context window size
    #[arg(long = "window-size", default_value_t = 15)]
    window_size: usize,

    /// Path to the vocabulary file
    #[arg(long = "vocab-file", default_value = "vocab.txt")]
    vocab_file: String,

    /// Use distance weighting
    #[arg(long = "distance-weighting", default_value_t = true)]
    distance_weighting: bool,

    /// Limit memory usage (in GB); 0.0 for 'unlimited'
    #[arg(long, default_value_t = 8.0)]
    memory: f64,

    /// File head for overflow files
    #[arg(long = "overflow-file", default_value = "overflow")]
    file_head: String,
}

// Configuration struct to replace global variables
pub struct Config {
    pub window_size: usize,
    pub verbose: u8,
    pub symmetric: bool,
    pub distance_weighting: bool,
    pub max_product: u64,
    pub overflow_length: usize,
    pub vocab_file: String,
    pub file_head: String,
}

// Wrapper for the priority queue, to keep track of the file source.
// Equivalent to CRECID.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct CrecId {
    crec: Crec,
    id: usize, // file index
}

// We implement Ord for CrecId to make the BinaryHeap a min-heap on `crec`.
// BinaryHeap is a max-heap, so we invert the comparison logic.
impl Ord for CrecId {
    fn cmp(&self, other: &Self) -> Ordering {
        other.crec.cmp(&self.crec)
    }
}

impl PartialOrd for CrecId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Collect word-word cooccurrence counts from stdin.
pub fn get_cooccurrences(config: &Config) -> io::Result<usize> {
    eprintln!("COUNTING COOCCURRENCES");
    if config.verbose > 0 {
        eprintln!("window size: {}", config.window_size);
        eprintln!(
            "context: {}",
            if config.symmetric {
                "symmetric"
            } else {
                "asymmetric"
            }
        );
    }
    if config.verbose > 1 {
        eprintln!("max product: {}", config.max_product);
        eprintln!("overflow length: {}", config.overflow_length);
        eprint!("Reading vocab from file \"{}\"...", config.vocab_file);
    }

    // --- Read Vocabulary ---
    let vocab = Vocabulary::from_file(config.vocab_file.as_ref())?;
    let vocab_size = vocab.size();
    if config.verbose > 1 {
        eprintln!("Loaded {vocab_size} words.");
    }
    if vocab_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty vocabulary",
        ));
    }

    // --- Build Lookup Table for co-occurances
    // lookup[i] is the start index in bigram_table for row i (0-based).
    // lookup[vocab_size] is one past the end of the last row.
    let mut lookup = vec![0usize; vocab_size + 1];
    lookup[0] = 0;
    for i in 0..vocab_size {
        let val = config.max_product / (i as u64 + 1);
        let allowed_cols = if val < vocab_size as u64 {
            val as usize
        } else {
            vocab_size
        };
        lookup[i + 1] = lookup[i] + allowed_cols;
    }
    if config.verbose > 1 {
        eprintln!("Lookup table contains {} elements.", lookup[vocab_size]);
    }

    // --- Allocate Memory ---
    let mut bigram_table = vec![0.0f64; lookup[vocab_size] as usize];
    let mut cr_overflow = Vec::with_capacity(config.overflow_length + 1);
    let mut history = vec![u32::MAX; config.window_size];
    let mut fid_counter = 1;

    let overflow_threshold = if config.symmetric {
        config
            .overflow_length
            .saturating_sub(2 * config.window_size)
    } else {
        config.overflow_length.saturating_sub(config.window_size)
    };

    // --- Process Tokens from Stdin ---
    if config.verbose > 1 {
        eprint!("Processing token: 0");
        io::stderr().flush()?;
    }

    let mut overflow_filename = format!("{}_{:04}.bin", config.file_head, fid_counter);
    let mut foverflow = BufWriter::new(File::create(&overflow_filename)?);

    let stdin = io::stdin();
    let mut token_counter = 0u64;

    let mut reader = BufReader::new(stdin.lock());
    let mut line = String::new();

    while reader.read_line(&mut line)? > 0 {
        let mut line_word_idx = 0usize;
        for word in line.split_whitespace() {
            if cr_overflow.len() >= overflow_threshold {
                // Sort and write the overflow buffer
                cr_overflow.sort_unstable();
                for rec in &cr_overflow {
                    Crec::write_to_raw(&mut foverflow, rec)?;
                }
                foverflow.flush()?;
                cr_overflow.clear();

                fid_counter += 1;
                overflow_filename = format!("{}_{:04}.bin", config.file_head, fid_counter);
                foverflow = BufWriter::new(File::create(&overflow_filename)?);
            }

            token_counter += 1;
            if config.verbose > 1 && token_counter % 100_000 == 0 {
                eprint!("\x1B[19G{token_counter}"); // ANSI escape to move cursor
                io::stderr().flush()?;
            }

            if let Some(&w2) = vocab.get_index(word) {
                // Iterate over context words in the history window
                let window_start = line_word_idx.saturating_sub(config.window_size);
                for k in (window_start..line_word_idx).rev() {
                    let w1 = history[k % config.window_size];
                    if w1 == u32::MAX {
                        continue;
                    }
                    let distance = (line_word_idx - k) as f64;
                    let weight = if config.distance_weighting {
                        1.0 / distance
                    } else {
                        1.0
                    };

                    // Check if the product of indices fits in bigram_table
                    if (w1 as u64 + 1) < config.max_product / (w2 as u64 + 1) {
                        // The bigram_table is a flattened, jagged 2D array.
                        // lookup[w1] gives the start index for row w1.
                        // w2 is the column offset within that row.
                        let index1 = lookup[w1 as usize] + w2;
                        bigram_table[index1] += weight;
                        if config.symmetric {
                            let index2 = lookup[w2] + w1 as usize;
                            bigram_table[index2] += weight;
                        }
                    } else {
                        // Product is too big, store in overflow buffer
                        let mut cr = Crec {
                            word1: w1 as u32, // convert to 0-based index
                            word2: w2 as u32, // convert to 0-based index
                            val: weight,
                        };
                        cr_overflow.push(cr);
                        if config.symmetric {
                            std::mem::swap(&mut cr.word1, &mut cr.word2);
                            cr_overflow.push(cr);
                        }
                    }
                }
                history[line_word_idx % config.window_size] = w2 as u32;
                line_word_idx += 1;
            }
        }
        line.clear(); // Clear the line buffer for reuse
    }

    // --- Final Write-out ---
    if config.verbose > 1 {
        eprintln!("\x1B[0GProcessed {token_counter} tokens.");
    }
    cr_overflow.sort_unstable();
    for rec in &cr_overflow {
        Crec::write_to_raw(&mut foverflow, rec)?;
    }
    cr_overflow.clear();
    drop(foverflow); // Close the last overflow file

    // Write the main bigram_table to file `_0000.bin`
    let bigram_filename = format!("{}_0000.bin", config.file_head);
    let mut fbigram = BufWriter::new(File::create(bigram_filename)?);
    if config.verbose > 1 {
        eprint!("Writing cooccurrences to disk");
    }

    // Calculate the number of digits in the total count for padding output
    let width = vocab_size.to_string().len();

    for x in 0..vocab_size {
        if config.verbose > 1 && (x + 1) % (vocab_size / 100 + 1) == 0 {
            eprint!("\rWriting cooccurrences: {x:>width$}/{vocab_size}");
            io::stderr().flush()?;
        }
        let y_limit = lookup[x + 1] - lookup[x];
        for y in 0..y_limit {
            let val = bigram_table[(lookup[x] + y) as usize];
            if val != 0.0 {
                Crec::write_to_raw(
                    &mut fbigram,
                    &Crec {
                        word1: x as u32,
                        word2: y as u32,
                        val,
                    },
                )?;
            }
        }
    }
    if config.verbose > 1 {
        eprintln!("\n{fid_counter} overflow files in total.");
    }

    // Return the total number of files to merge (main bigram file + overflow files)
    Ok(fid_counter + 1)
}

/// Merge `num` sorted files of cooccurrence records into a single stream on stdout.
pub fn merge_files(config: &Config, num_files: usize) -> io::Result<()> {
    let mut file_readers: Vec<Option<BufReader<File>>> = Vec::with_capacity(num_files);
    let mut pq = BinaryHeap::with_capacity(num_files); // Our min-heap
    let mut filenames = Vec::with_capacity(num_files);

    if config.verbose > 1 {
        eprint!("Merging cooccurrence files: processed 0 records.");
        io::stderr().flush()?;
    }

    // --- Open all files and populate the priority queue ---
    for i in 0..num_files {
        let filename = format!("{}_{:04}.bin", config.file_head, i);
        let file = match File::open(&filename) {
            Ok(f) => f,
            Err(e) => {
                // If a file can't be opened (e.g., _0000.bin was empty), just warn and skip.
                eprintln!("\nWarning: Could not open temp file {filename}: {e}. Skipping.");
                file_readers.push(None); // Add a placeholder
                continue;
            }
        };
        filenames.push(filename);
        let mut reader = BufReader::new(file);
        if let Some(crec) = Crec::read_from_raw(&mut reader)? {
            pq.push(CrecId { crec, id: i });
        }
        file_readers.push(Some(reader));
    }

    let mut fout = BufWriter::new(io::stdout());
    let mut counter = 0i64;

    // --- Merge records using the priority queue ---

    // 1. Pop the first element to initialize the accumulator
    let Some(mut old_item) = pq.pop() else {
        eprintln!("\nNo data to merge.");
        return Ok(());
    };

    // 2. Refill the queue from the file we just read from.
    let file_id = old_item.id;
    if let Some(Some(reader)) = file_readers.get_mut(file_id) {
        if let Some(new_crec) = Crec::read_from_raw(reader)? {
            pq.push(CrecId {
                crec: new_crec,
                id: file_id,
            });
        }
    }

    // 3. Process the rest of the queue
    while let Some(new_item) = pq.pop() {
        if new_item.crec == old_item.crec {
            old_item.crec.val += new_item.crec.val;
        } else {
            // Different word pair. Write the completed `old` record...
            Crec::write_to_raw(&mut fout, &old_item.crec)?;
            counter += 1;
            if config.verbose > 1 && counter % 100_000 == 0 {
                eprint!("\x1B[39G{counter} records.");
                io::stderr().flush()?;
            }
            // ...and start accumulating the new one.
            old_item = new_item;
        }

        // Refill the queue from the file of the item we just processed.
        let file_id = new_item.id;
        if let Some(Some(reader)) = file_readers.get_mut(file_id) {
            if let Some(next_crec) = Crec::read_from_raw(reader)? {
                pq.push(CrecId {
                    crec: next_crec,
                    id: file_id,
                });
            }
        }
    }

    // 4. Write the very last accumulated record.
    Crec::write_to_raw(&mut fout, &old_item.crec)?;
    counter += 1;
    fout.flush()?;

    eprintln!("\x1B[0GMerging cooccurrence files: processed {counter} records.");

    // --- Cleanup: Remove temporary files ---
    for filename in filenames {
        if let Err(e) = fs::remove_file(&filename) {
            eprintln!("Warning: could not remove temp file {filename}: {e}");
        }
    }
    eprintln!("\n");

    Ok(())
}

/// Calculates max_product and overflow_length based on a memory limit in GB.
fn calculate_memory_params(memory_limit_gb: f64) -> (u64, usize) {
    if memory_limit_gb <= 0.0 {
        // use very large values to simulate "unlimited".
        const LARGE_MAX_PRODUCT: u64 = 1_000_000_000; // 1 billion
        const LARGE_OVERFLOW_LENGTH: usize = 1_000_000_000; // 1 billion
        return (LARGE_MAX_PRODUCT, LARGE_OVERFLOW_LENGTH);
    }

    const GIGABYTE: f64 = 1_073_741_824.0; // bytes, ie. 1024*1024*1024
    const MEMORY_UTILIZATION_FACTOR: f64 = 0.85; // For main hash table
    const OVERFLOW_MEMORY_FACTOR: f64 = 1.0 - MEMORY_UTILIZATION_FACTOR; // For overflow buffer
    const NEWTON_RAPHSON_CONSTANT: f64 = 0.1544313298; // From original C code

    // Calculate rlimit: the target number of CREC elements that can fit in 85% of the memory.
    let crec_size = mem::size_of::<Crec>() as f64;
    let total_memory_in_bytes = memory_limit_gb * GIGABYTE;
    let rlimit = MEMORY_UTILIZATION_FACTOR * total_memory_in_bytes / crec_size;

    // Solve for n using Newton-Raphson-like iterative method.
    // n is an estimate for max_product.
    // Start with an initial guess for n. rlimit / log(rlimit) is a reasonable start.
    let mut n = rlimit / rlimit.ln();

    for _ in 0..100 {
        // Use a fixed number of iterations to prevent infinite loops
        let next_n = rlimit / (n.ln() + NEWTON_RAPHSON_CONSTANT);
        if (n - next_n).abs() < 1e-3 {
            break; // Converged
        }
        n = next_n;
    }

    let max_product = n as u64;
    let overflow_length = (OVERFLOW_MEMORY_FACTOR * total_memory_in_bytes / crec_size) as usize;
    (max_product, overflow_length)
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let (max_product, overflow_length) = calculate_memory_params(args.memory);

    if args.window_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Window size must be positive",
        ));
    }

    let config = Config {
        verbose: args.verbose,
        symmetric: !args.asymmetric,
        window_size: args.window_size,
        vocab_file: args.vocab_file,
        file_head: args.file_head,
        distance_weighting: args.distance_weighting,
        max_product,
        overflow_length,
    };

    let num_tmp_files = get_cooccurrences(&config)?;

    // Merge the temporary files into the final output on stdout.
    if num_tmp_files > 0 {
        merge_files(&config, num_tmp_files)?;
    } else {
        eprintln!("No cooccurrence files were generated to merge.");
    }

    Ok(())
}
