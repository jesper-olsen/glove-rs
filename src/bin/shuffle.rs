//! Tool to shuffle entries of word-word cooccurrence files
//!

use glove_rs::Crec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Stderr, Write};
use std::mem;
use std::process;
use std::time::SystemTime;

// Global configuration parameters, grouped into a struct for better organization
// than global static variables.
struct Config {
    verbose: i32,
    seed: Option<u64>,
    array_size: usize,
    temp_file_head: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Handle help flag
    if args.len() == 2 && (args[1] == "-h" || args[1] == "-help" || args[1] == "--help") {
        print_usage();
        return;
    }

    // Parse arguments and build config
    let config = match Config::from_args(&args) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Argument parsing error: {}", e);
            print_usage();
            process::exit(1);
        }
    };

    // Run the main shuffling logic
    if let Err(e) = shuffle_by_chunks(config) {
        eprintln!("\nAn error occurred: {}", e);
        process::exit(1);
    }
}

impl Config {
    /// Parses command-line arguments to build a Config struct.
    fn from_args(args: &[String]) -> Result<Self, String> {
        let mut verbose = 2;
        let mut temp_file_head = "temp_shuffle".to_string();
        let mut memory_limit = 2.0;
        let mut array_size: Option<usize> = None;
        let mut seed: Option<u64> = None;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "-verbose" => {
                    i += 1;
                    verbose = args
                        .get(i)
                        .ok_or("-verbose requires a value")?
                        .parse()
                        .map_err(|e| format!("Invalid verbose value: {}", e))?;
                }
                "-temp-file" => {
                    i += 1;
                    temp_file_head = args.get(i).ok_or("-temp-file requires a value")?.clone();
                }
                "-memory" => {
                    i += 1;
                    memory_limit = args
                        .get(i)
                        .ok_or("-memory requires a value")?
                        .parse()
                        .map_err(|e| format!("Invalid memory value: {}", e))?;
                }
                "-array-size" => {
                    i += 1;
                    array_size = Some(
                        args.get(i)
                            .ok_or("-array-size requires a value")?
                            .parse()
                            .map_err(|e| format!("Invalid array-size value: {}", e))?,
                    );
                }
                "-seed" => {
                    i += 1;
                    seed = Some(
                        args.get(i)
                            .ok_or("-seed requires a value")?
                            .parse()
                            .map_err(|e| format!("Invalid seed value: {}", e))?,
                    );
                }
                _ => return Err(format!("Unknown argument: {}", args[i])),
            }
            i += 1;
        }

        let final_array_size = array_size.unwrap_or_else(|| {
            // 1 GiB = 1073741824 bytes
            (0.95 * memory_limit * 1_073_741_824.0 / mem::size_of::<Crec>() as f64) as usize
        });

        Ok(Config {
            verbose,
            seed,
            array_size: final_array_size,
            temp_file_head,
        })
    }
}

/// Main logic: reads stdin in chunks, shuffles them, saves them to temp files,
/// and then merges the shuffled temp files.
fn shuffle_by_chunks(config: Config) -> io::Result<()> {
    let mut stderr = io::stderr();
    let seed = config.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });

    writeln!(stderr, "Using random seed {}", seed)?;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut array: Vec<Crec> = Vec::with_capacity(config.array_size);
    let mut stdin = io::stdin().lock();
    let mut file_counter = 0;
    let mut total_lines: u64 = 0;

    writeln!(stderr, "SHUFFLING COOCCURRENCES")?;
    if config.verbose > 0 {
        writeln!(stderr, "array size: {}", config.array_size)?;
    }
    if config.verbose > 1 {
        write!(stderr, "Shuffling by chunks: processed 0 lines.")?;
        stderr.flush()?;
    }

    loop {
        // Read one Crec struct from stdin
        let mut crec_buf = [0u8; mem::size_of::<Crec>()];
        match stdin.read_exact(&mut crec_buf) {
            Ok(_) => {
                // This is safe because we read exactly size_of::<Crec>() bytes and Crec is #[repr(C)].
                let crec: Crec = unsafe { mem::transmute(crec_buf) };
                array.push(crec);
            }
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                // End of file, break the loop to process the final chunk.
                break;
            }
            Err(e) => return Err(e),
        }

        // If the array is full, shuffle it and write to a temporary file.
        if array.len() >= config.array_size {
            total_lines += array.len() as u64;
            shuffle_and_write_chunk(
                &mut array,
                &mut rng,
                &config,
                file_counter,
                total_lines,
                &mut stderr,
            )?;
            file_counter += 1;
            array.clear();
        }
    }

    // Process the final, potentially smaller, chunk.
    if !array.is_empty() {
        total_lines += array.len() as u64;
        shuffle_and_write_chunk(
            &mut array,
            &mut rng,
            &config,
            file_counter,
            total_lines,
            &mut stderr,
        )?;
        file_counter += 1;
    }

    if config.verbose > 1 {
        // Overwrite the progress line
        write!(
            stderr,
            "\x1B[2K\rShuffling by chunks: processed {} lines.\n",
            total_lines
        )?;
        writeln!(stderr, "Wrote {} temporary file(s).", file_counter)?;
    }

    // Merge the temporary files
    shuffle_merge(file_counter, &config, &mut rng)
}

/// Helper to shuffle a chunk and write it to a temporary file.
fn shuffle_and_write_chunk(
    array: &mut Vec<Crec>,
    rng: &mut StdRng,
    config: &Config,
    file_id: usize,
    total_lines: u64,
    stderr: &mut Stderr,
) -> io::Result<()> {
    // Fisher-Yates shuffle provided by the `rand` crate.
    // The C code shuffles up to `i-2`, which seems like a potential off-by-one error.
    // A standard shuffle shuffles the whole array. We will shuffle the whole slice.
    array.shuffle(rng);

    if config.verbose > 1 {
        write!(
            stderr,
            "\rShuffling by chunks: processed {} lines.",
            total_lines
        )?;
        stderr.flush()?;
    }

    let filename = format!("{}_{:04}.bin", config.temp_file_head, file_id);
    let file = File::create(&filename)?;
    let mut writer = BufWriter::new(file);

    // Write the contents of the array to the binary file.
    // This is safe because Crec is a "plain old data" (POD) type.
    let byte_slice = unsafe {
        std::slice::from_raw_parts(
            array.as_ptr() as *const u8,
            array.len() * mem::size_of::<Crec>(),
        )
    };
    writer.write_all(byte_slice)?;

    Ok(())
}

/// Merges shuffled temporary files into stdout.
fn shuffle_merge(num_files: usize, config: &Config, rng: &mut StdRng) -> io::Result<()> {
    if num_files == 0 {
        return Ok(());
    }

    let mut stderr = io::stderr();
    let mut stdout = BufWriter::new(io::stdout().lock());
    let mut total_lines: u64 = 0;

    // Open all temporary files for reading
    let mut readers = Vec::new();
    for i in 0..num_files {
        let filename = format!("{}_{:04}.bin", config.temp_file_head, i);
        let file = File::open(&filename)?;
        readers.push(BufReader::new(file));
    }

    if config.verbose > 0 {
        write!(stderr, "Merging temp files: processed 0 lines.")?;
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
                        // This is safe because Crec is #[repr(C)] and we read the exact size.
                        let crec: Crec = unsafe { mem::transmute(crec_buf) };
                        merge_buffer.push(crec);
                        any_data_read = true;
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                        // This file is exhausted; stop reading from it for this pass.
                        break;
                    }
                    Err(e) => return Err(e), // Propagate other I/O errors.
                }
            }
        }

        // If no data was read from any file, all files are exhausted, so we're done.
        if !any_data_read {
            break;
        }

        total_lines += merge_buffer.len() as u64;

        // Shuffle the merged buffer. The `rand` crate provides a robust implementation.
        merge_buffer.shuffle(rng);

        // Write the shuffled buffer to stdout.
        let byte_slice = unsafe {
            std::slice::from_raw_parts(
                merge_buffer.as_ptr() as *const u8,
                merge_buffer.len() * mem::size_of::<Crec>(),
            )
        };
        stdout.write_all(byte_slice)?;

        if config.verbose > 0 {
            // Use ANSI escape codes to update the line in place.
            write!(stderr, "\x1B[31G{} lines.", total_lines)?;
            stderr.flush()?;
        }
    }

    if config.verbose > 0 {
        // Overwrite the progress line with the final count and move to a new line.
        writeln!(
            stderr,
            "\x1B[0GMerging temp files: processed {} lines.",
            total_lines
        )?;
    }

    // Clean up: close and remove temporary files.
    // File handles in `readers` are closed automatically when they go out of scope.
    // We just need to remove the files from disk.
    for i in 0..num_files {
        let filename = format!("{}_{:04}.bin", config.temp_file_head, i);
        if let Err(e) = fs::remove_file(&filename) {
            // Log a warning but don't fail the entire program.
            eprintln!("Warning: could not remove temp file '{}': {}", filename, e);
        }
    }

    writeln!(stderr, "\n")?;

    Ok(())
}

/// Prints the command-line usage instructions to stderr.
fn print_usage() {
    eprintln!("Tool to shuffle entries of word-word cooccurrence files");
    eprintln!("Rust port of the original C code by Jeffrey Pennington (jpennin@stanford.edu)\n");
    eprintln!("Usage options:");
    eprintln!("\t-verbose <int>");
    eprintln!("\t\tSet verbosity: 0, 1, or 2 (default)");
    eprintln!("\t-memory <float>");
    eprintln!("\t\tSoft limit for memory consumption, in GB; default 2.0");
    eprintln!("\t-array-size <int>");
    eprintln!(
        "\t\tLimit to length <int> the buffer which stores chunks of data to shuffle before writing to disk."
    );
    eprintln!("\t\tThis value overrides that which is automatically produced by '-memory'.");
    eprintln!("\t-temp-file <file>");
    eprintln!("\t\tFilename, excluding extension, for temporary files; default temp_shuffle");
    eprintln!("\t-seed <int>");
    eprintln!("\t\tRandom seed to use. If not set, will be randomized using current time.");
    eprintln!("\nExample usage: (assuming 'cooccurrence.bin' has been produced by 'cooccur')");
    eprintln!("./shuffle -verbose 2 -memory 8.0 < cooccurrence.bin > cooccurrence.shuf.bin");
}
