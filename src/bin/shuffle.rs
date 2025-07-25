//! Tool to shuffle entries of word-word cooccurrence files
//!

use clap::Parser;
use glove_rs::Crec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Stderr, Write};
use std::mem;
use std::process;
use std::time::SystemTime;

/// Tool to shuffle entries of word-word cooccurrence files
#[derive(Parser, Debug)]
#[command(version, about, long_about = None, verbatim_doc_comment)]
struct Cli {
    /// Set verbosity: 0, 1, or 2
    #[arg(long = "verbose", default_value_t = 2)]
    verbose: i32,

    /// Soft limit for memory consumption, in GB
    #[arg(long = "memory", default_value_t = 2.0)]
    memory: f64,

    /// Limit the buffer size (overrides the calculation from --memory)
    #[arg(long = "array-size")]
    array_size: Option<usize>,

    /// Filename, excluding extension, for temporary files
    #[arg(long = "temp-file", default_value = "temp_shuffle")]
    temp_file: String,

    /// Random seed to use. If not set, a random seed is generated.
    #[arg(long = "seed")]
    seed: Option<u64>,
}

struct Config {
    verbose: i32,
    seed: Option<u64>,
    array_size: usize,
    temp_file_head: String,
}

fn main() {
    let cli = Cli::parse(); // clap handles all parsing and validation

    let final_array_size = cli.array_size.unwrap_or_else(|| {
        const ONE_GB: f64 = 1_073_741_824.0;
        (0.95 * cli.memory * ONE_GB / mem::size_of::<Crec>() as f64) as usize
    });

    let config = Config {
        verbose: cli.verbose,
        seed: cli.seed,
        array_size: final_array_size,
        temp_file_head: cli.temp_file,
    };

    if let Err(e) = shuffle_by_chunks(config) {
        eprintln!("\nAn error occurred: {e}");
        process::exit(1);
    }
}

/// reads stdin in chunks, shuffles them, saves them to temp files, and then merges the shuffled temp files.
fn shuffle_by_chunks(config: Config) -> io::Result<()> {
    let mut stderr = io::stderr();
    let seed = config.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    writeln!(stderr, "Using random seed {seed}")?;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut array: Vec<Crec> = Vec::with_capacity(config.array_size);
    let mut stdin = io::stdin().lock();
    let mut file_counter = 0;
    let mut total_records: u64 = 0;

    writeln!(stderr, "Shuffling cooccurrences")?;
    if config.verbose > 0 {
        writeln!(stderr, "array size: {}", config.array_size)?;
    }
    if config.verbose > 1 {
        write!(stderr, "Shuffling by chunks: processed 0 records.")?;
        stderr.flush()?;
    }

    while let Some(crec) = Crec::read_from_raw(&mut stdin)? {
        array.push(crec);
        // If the array is full, shuffle it and write to a temporary file.
        if array.len() >= config.array_size {
            total_records += array.len() as u64;
            array.shuffle(&mut rng); // Fisher-Yates shuffle provided by the `rand` crate.
            save_shuffled_chunk(
                &mut array,
                &config,
                file_counter,
                total_records,
                &mut stderr,
            )?;
            file_counter += 1;
            array.clear();
        }
    }

    // Process the final, potentially smaller, chunk.
    if !array.is_empty() {
        total_records += array.len() as u64;
        array.shuffle(&mut rng);
        save_shuffled_chunk(
            &mut array,
            &config,
            file_counter,
            total_records,
            &mut stderr,
        )?;
        file_counter += 1;
    }

    if config.verbose > 1 {
        // Overwrite the progress line
        write!(
            stderr,
            "\x1B[2K\rShuffling by chunks: processed {total_records} records.\n"
        )?;
        writeln!(stderr, "Wrote {file_counter} temporary file(s).")?;
    }

    shuffle_merge(file_counter, &config, &mut rng)
}

/// Helper to shuffle a chunk and write it to a temporary file.
fn save_shuffled_chunk(
    array: &mut [Crec],
    config: &Config,
    file_id: usize,
    total_records: u64,
    stderr: &mut Stderr,
) -> io::Result<()> {
    if config.verbose > 1 {
        write!(
            stderr,
            "\rShuffling by chunks: processed {total_records} records."
        )?;
        stderr.flush()?;
    }

    let filename = format!("{x}_{file_id:04}.bin", x = config.temp_file_head);
    let file = File::create(&filename)?;
    let mut writer = BufWriter::new(file);
    Crec::write_slice_raw(&mut writer, array)?;

    Ok(())
}

/// Merges shuffled temporary files into stdout.
fn shuffle_merge(num_files: usize, config: &Config, rng: &mut StdRng) -> io::Result<()> {
    if num_files == 0 {
        return Ok(());
    }

    let mut stderr = io::stderr();
    let mut stdout = BufWriter::new(io::stdout().lock());
    let mut total_records: u64 = 0;

    // Open all temporary files for reading
    let mut readers = Vec::new();
    for i in 0..num_files {
        let filename = format!("{}_{:04}.bin", config.temp_file_head, i);
        let file = File::open(&filename)?;
        readers.push(BufReader::new(file));
    }

    if config.verbose > 0 {
        write!(stderr, "Merging temp files: processed 0 records.")?;
        stderr.flush()?;
    }

    let mut merge_buffer: Vec<Crec> = Vec::with_capacity(config.array_size);
    let mut crec_buf = [0u8; mem::size_of::<Crec>()];

    // Determine how many records to read from each temp file in each merge pass.
    let chunk_per_file = if num_files > 0 {
        (config.array_size / num_files).max(1) // Ensure we read at least one record
    } else {
        config.array_size
    };

    loop {
        merge_buffer.clear();
        let mut any_data_read = false;

        // Read a chunk from each temp file into the merge buffer.
        for reader in readers.iter_mut() {
            for _ in 0..chunk_per_file {
                match reader.read_exact(&mut crec_buf) {
                    Ok(_) => {
                        let crec: Crec = unsafe { mem::transmute(crec_buf) };
                        merge_buffer.push(crec);
                        any_data_read = true;
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // If no data was read from any file, all files are exhausted, so we're done.
        if !any_data_read {
            break;
        }

        total_records += merge_buffer.len() as u64;
        merge_buffer.shuffle(rng);

        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                merge_buffer.as_ptr() as *const u8,
                merge_buffer.len() * mem::size_of::<Crec>(),
            )
        };
        stdout.write_all(byte_slice)?;

        if config.verbose > 0 {
            write!(stderr, "\x1B[31G{total_records} records.")?;
            stderr.flush()?;
        }
    }

    if config.verbose > 0 {
        writeln!(
            stderr,
            "\x1B[0GMerging temp files: processed {total_records} records."
        )?;
    }

    // Clean up temporary files.
    for i in 0..num_files {
        let filename = format!("{}_{:04}.bin", config.temp_file_head, i);
        if let Err(e) = fs::remove_file(&filename) {
            eprintln!("Warning: could not remove temp file '{filename}': {e}");
        }
    }

    writeln!(stderr, "\n")?;

    Ok(())
}
