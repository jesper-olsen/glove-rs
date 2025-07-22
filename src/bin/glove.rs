use bytemuck::pod_read_unaligned;
use chrono::Local;
use clap::Parser;
use glove_rs::Crec;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::error::Error;
use std::fs::{File, metadata};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use std::{mem, thread};

/// Command-line arguments parsed by Clap.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_parser, default_value_t = 2)]
    verbose: i32,
    #[clap(long, value_parser, default_value_t = 50)]
    vector_size: usize,
    #[clap(long, value_parser, default_value_t = 15)]
    iter: usize,
    #[clap(long, value_parser, default_value_t = 8)]
    threads: usize,
    #[clap(long, value_parser, default_value_t = 0.05)]
    eta: f64,
    #[clap(long, value_parser, default_value_t = 0.75)]
    alpha: f64,
    #[clap(long, value_parser, default_value_t = 10.0)]
    x_max: f64,
    #[clap(
        long,
        value_parser,
        default_value_t = 2,
        help = "0: text, 1: binary, 2: both"
    )]
    binary: i32,
    #[clap(
        long,
        value_parser,
        default_value_t = 2,
        help = "0: all, 1: w, 2: w+w', 3: w,w'"
    )]
    model: i32,
    #[clap(long, value_parser, required = true)]
    input_file: PathBuf,
    #[clap(long, value_parser, required = true)]
    vocab_file: PathBuf,
    #[clap(long, value_parser, required = true)]
    save_file: PathBuf,
    #[clap(long, value_parser, default_value_t = 0)]
    seed: u64,
}

/// Configuration parameters, built from command-line arguments.
#[derive(Debug, Clone)]
struct Config {
    vocab_file: PathBuf,
    input_file: PathBuf,
    save_file: PathBuf,
    vector_size: usize,
    num_threads: usize,
    num_iter: usize,
    use_binary: i32,
    model: i32,
    x_max: f64,
    alpha: f64,
    eta: f64,
    grad_clip_value: f64,
    verbose: i32,
    seed: u64,
}

/// A wrapper around a Vec that allows it to be shared across threads unsafely.
/// This is necessary to replicate the lock-free SGD ("Hogwild!") behavior of
/// the original C code, where data races on vector updates are tolerated.
struct UnsafeSyncVec(Vec<f64>);
unsafe impl Sync for UnsafeSyncVec {}

/// Holds the model parameters (word vectors and their squared gradients).
struct GloveModel {
    w: Arc<UnsafeSyncVec>,
    gradsq: Arc<UnsafeSyncVec>,
}

/// Entry point for the training process.
fn train_glove(config: &Config) -> Result<(), Box<dyn Error>> {
    eprintln!("TRAINING MODEL");

    let vocab_size = count_vocab(&config.vocab_file)?;
    let num_lines = {
        let file_meta = metadata(&config.input_file)?;
        (file_meta.len() / mem::size_of::<Crec>() as u64) as usize
    };

    if config.verbose > 1 {
        eprintln!("Read {} lines.", num_lines);
        eprintln!("Initializing parameters...");
    }

    let model = initialize_parameters(config, vocab_size);
    let total_cost = Arc::new(Mutex::new(vec![0.0; config.num_threads]));

    if config.verbose > 1 {
        eprintln!("done.");
    }
    if config.verbose > 0 {
        eprintln!("vocab size: {}", vocab_size);
        eprintln!("vector size: {}", config.vector_size);
        eprintln!("x_max: {}", config.x_max);
        eprintln!("alpha: {}", config.alpha);
    }

    let lines_per_thread = calculate_lines_per_thread(num_lines, config.num_threads);

    for b in 0..config.num_iter {
        *total_cost.lock().unwrap() = vec![0.0; config.num_threads];

        thread::scope(|s| {
            for id in 0..config.num_threads {
                let thread_config = config.clone();
                let w_clone = Arc::clone(&model.w);
                let gradsq_clone = Arc::clone(&model.gradsq);
                let thread_cost_arc = Arc::clone(&total_cost);
                let thread_lines_count = lines_per_thread[id];
                let thread_offset = lines_per_thread.iter().take(id).sum::<usize>();

                s.spawn(move || {
                    glove_thread(
                        id,
                        &thread_config,
                        &w_clone,
                        &gradsq_clone,
                        &thread_cost_arc,
                        thread_lines_count,
                        thread_offset,
                        vocab_size,
                    );
                });
            }
        });

        let final_cost: f64 = total_cost.lock().unwrap().iter().sum();
        let time_str = Local::now().format("%x - %I:%M.%S%p");
        eprintln!(
            "{}, iter: {:03}, cost: {}",
            time_str,
            b + 1,
            final_cost / num_lines as f64
        );
    }

    eprintln!("Saving final parameters...");
    save_params(&model.w.0, config, vocab_size, -1)?;
    eprintln!("done.");

    Ok(())
}

fn glove_thread(
    id: usize,
    config: &Config,
    w_arc: &Arc<UnsafeSyncVec>,
    gradsq_arc: &Arc<UnsafeSyncVec>,
    cost_arc: &Arc<Mutex<Vec<f64>>>,
    lines_to_process: usize,
    file_offset: usize,
    vocab_size: usize,
) {
    let w_ptr = w_arc.0.as_ptr() as *mut f64;
    let gradsq_ptr = gradsq_arc.0.as_ptr() as *mut f64;

    let mut thread_cost = 0.0;
    let mut fin = match File::open(&config.input_file) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Thread {}: Failed to open input file: {}", id, e);
            return;
        }
    };

    let start_offset = file_offset * mem::size_of::<Crec>();
    if let Err(e) = fin.seek(SeekFrom::Start(start_offset as u64)) {
        eprintln!("Thread {}: Failed to seek in input file: {}", id, e);
        return;
    }

    let mut reader = BufReader::with_capacity(8192, fin);
    let mut corec_buf = [0u8; mem::size_of::<Crec>()];

    // FIX 2: Explicitly type the float literal to resolve ambiguity for the compiler.
    let mut w_updates1 = vec![0.0 as f64; config.vector_size];
    let mut w_updates2 = vec![0.0 as f64; config.vector_size];

    for _ in 0..lines_to_process {
        if reader.read_exact(&mut corec_buf).is_err() {
            break;
        }

        // FIX 1: Use pod_read_unaligned for a direct, unambiguous conversion from bytes to struct.
        let cr: Crec = pod_read_unaligned(&corec_buf);

        if cr.word1 < 1 || cr.word2 < 1 {
            continue;
        }

        let l1 = (cr.word1 as usize - 1) * (config.vector_size + 1);
        let l2 = (cr.word2 as usize - 1 + vocab_size) * (config.vector_size + 1);

        unsafe {
            let mut diff = 0.0;
            for i in 0..config.vector_size {
                diff += *w_ptr.add(i + l1) * *w_ptr.add(i + l2);
            }
            diff += *w_ptr.add(config.vector_size + l1) + *w_ptr.add(config.vector_size + l2)
                - cr.val.ln();

            let fdiff = if cr.val > config.x_max {
                diff
            } else {
                (cr.val / config.x_max).powf(config.alpha) * diff
            };

            if diff.is_nan() || fdiff.is_nan() {
                continue;
            }

            thread_cost += 0.5 * fdiff * diff;

            for i in 0..config.vector_size {
                let grad1 = fdiff * *w_ptr.add(i + l2);
                let grad2 = fdiff * *w_ptr.add(i + l1);
                let temp1 =
                    grad1.clamp(-config.grad_clip_value, config.grad_clip_value) * config.eta;
                let temp2 =
                    grad2.clamp(-config.grad_clip_value, config.grad_clip_value) * config.eta;
                w_updates1[i] = temp1 / (*gradsq_ptr.add(i + l1)).sqrt();
                w_updates2[i] = temp2 / (*gradsq_ptr.add(i + l2)).sqrt();
                *gradsq_ptr.add(i + l1) += temp1 * temp1;
                *gradsq_ptr.add(i + l2) += temp2 * temp2;
            }

            if !w_updates1.iter().any(|&v| v.is_nan() || v.is_infinite())
                && !w_updates2.iter().any(|&v| v.is_nan() || v.is_infinite())
            {
                for i in 0..config.vector_size {
                    *w_ptr.add(i + l1) -= w_updates1[i];
                    *w_ptr.add(i + l2) -= w_updates2[i];
                }
            }

            let bias_grad = fdiff * config.eta;
            let update1 = bias_grad / (*gradsq_ptr.add(config.vector_size + l1)).sqrt();
            let update2 = bias_grad / (*gradsq_ptr.add(config.vector_size + l2)).sqrt();
            *w_ptr.add(config.vector_size + l1) -= check_nan(update1);
            *w_ptr.add(config.vector_size + l2) -= check_nan(update2);
            let fdiff_sq = fdiff * fdiff;
            *gradsq_ptr.add(config.vector_size + l1) += fdiff_sq;
            *gradsq_ptr.add(config.vector_size + l2) += fdiff_sq;
        }
    }

    cost_arc.lock().unwrap()[id] = thread_cost;
}

fn initialize_parameters(config: &Config, vocab_size: usize) -> GloveModel {
    let mut rng = if config.seed == 0 {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        eprintln!("Using random seed {}", seed);
        StdRng::seed_from_u64(seed)
    } else {
        eprintln!("Using random seed {}", config.seed);
        StdRng::seed_from_u64(config.seed)
    };
    let w_size = 2 * vocab_size * (config.vector_size + 1);
    let mut w_vec = vec![0.0 as f64; w_size];
    for val in w_vec.iter_mut() {
        *val = (rng.random::<f64>() - 0.5) / config.vector_size as f64;
    }
    // FIX 2: Explicitly type the float literal here as well.
    let gradsq_vec = vec![1.0 as f64; w_size];
    GloveModel {
        w: Arc::new(UnsafeSyncVec(w_vec)),
        gradsq: Arc::new(UnsafeSyncVec(gradsq_vec)),
    }
}

fn save_params(w: &[f64], config: &Config, vocab_size: usize, iter: i32) -> io::Result<()> {
    let save_file_str = config.save_file.to_str().unwrap();

    if config.use_binary > 0 {
        let bin_filename = if iter < 0 {
            format!("{}.bin", save_file_str)
        } else {
            format!("{}.{:03}.bin", save_file_str, iter)
        };
        let mut f_out = BufWriter::new(File::create(bin_filename)?);
        f_out.write_all(bytemuck::cast_slice(w))?;
    }

    if config.use_binary != 1 {
        let txt_filename = if iter < 0 {
            format!("{}.txt", save_file_str)
        } else {
            format!("{}.{:03}.txt", save_file_str, iter)
        };
        let mut f_out = BufWriter::new(File::create(txt_filename)?);
        let vocab_file = BufReader::new(File::open(&config.vocab_file)?);

        let vocab: Vec<String> = vocab_file
            .lines()
            .map(|line| line.unwrap().split_whitespace().next().unwrap().to_string())
            .collect();

        for (i, word) in vocab.iter().enumerate().take(vocab_size) {
            write!(f_out, "{}", word)?;
            let l1 = i * (config.vector_size + 1);
            let l2 = (i + vocab_size) * (config.vector_size + 1);

            match config.model {
                0 => {
                    // Save all parameters
                    for j in 0..=config.vector_size {
                        write!(f_out, " {}", w[l1 + j])?;
                    }
                    for j in 0..=config.vector_size {
                        write!(f_out, " {}", w[l2 + j])?;
                    }
                }
                1 => {
                    // Save word vectors
                    for j in 0..config.vector_size {
                        write!(f_out, " {}", w[l1 + j])?;
                    }
                }
                2 => {
                    // Save word + context vectors
                    for j in 0..config.vector_size {
                        write!(f_out, " {}", w[l1 + j] + w[l2 + j])?;
                    }
                }
                3 => {
                    // Save word and context vectors concatenated
                    for j in 0..config.vector_size {
                        write!(f_out, " {}", w[l1 + j])?;
                    }
                    for j in 0..config.vector_size {
                        write!(f_out, " {}", w[l2 + j])?;
                    }
                }
                _ => {}
            }
            writeln!(f_out)?;
        }
    }
    Ok(())
}

fn count_vocab(path: &Path) -> io::Result<usize> {
    Ok(BufReader::new(File::open(path)?).lines().count())
}

fn calculate_lines_per_thread(num_lines: usize, num_threads: usize) -> Vec<usize> {
    if num_threads == 0 {
        return vec![];
    }
    let mut lines_per_thread = vec![num_lines / num_threads; num_threads];
    for i in 0..(num_lines % num_threads) {
        lines_per_thread[i] += 1;
    }
    lines_per_thread
}

#[inline]
fn check_nan(update: f64) -> f64 {
    if update.is_nan() || update.is_infinite() {
        0.0
    } else {
        update
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let config = Config {
        vocab_file: cli.vocab_file,
        input_file: cli.input_file,
        save_file: cli.save_file,
        vector_size: cli.vector_size,
        num_threads: cli.threads,
        num_iter: cli.iter,
        use_binary: cli.binary,
        model: cli.model,
        x_max: cli.x_max,
        alpha: cli.alpha,
        eta: cli.eta,
        grad_clip_value: 100.0, // Hardcoded as in original C
        verbose: cli.verbose,
        seed: cli.seed,
    };

    train_glove(&config)
}
