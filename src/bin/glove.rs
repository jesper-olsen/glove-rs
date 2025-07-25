use chrono::Local;
use clap::Parser;
use glove_rs::Crec;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::error::Error;
use std::fs::{File, metadata};
use std::io::{self, BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
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

/// Contains references to the shared model data.
struct SharedModel<'a> {
    w: &'a Arc<UnsafeSyncVec>,
    gradsq: &'a Arc<UnsafeSyncVec>,
    vocab_size: usize,
}

/// Defines the specific workload for a single thread.
struct ThreadTask {
    records_to_process: usize,
    file_offset: usize,
}

/// Entry point for the training process.
fn train_glove(config: &Config) -> Result<(), Box<dyn Error>> {
    eprintln!("Training GloVe Model");

    let vocab_size = count_vocab(&config.vocab_file)?;
    let num_records = {
        let file_meta = metadata(&config.input_file)?;
        (file_meta.len() / mem::size_of::<Crec>() as u64) as usize
    };

    if config.verbose > 1 {
        eprintln!("Read {num_records} records.");
    }

    let model = initialize_parameters(config, vocab_size);

    if config.verbose > 0 {
        eprintln!("vocab size: {vocab_size}");
        eprintln!("vector size: {}", config.vector_size);
        eprintln!("x_max: {}; alpha {}", config.x_max, config.alpha);
    }

    let records_per_thread = calculate_records_per_thread(num_records, config.num_threads);

    for b in 0..config.num_iter {
        let mut final_cost = 0.0;
        thread::scope(|s| {
            let mut handles = vec![];
            for id in 0..config.num_threads {
                let model_state = SharedModel {
                    w: &model.w,
                    gradsq: &model.gradsq,
                    vocab_size,
                };

                let task_spec = ThreadTask {
                    records_to_process: records_per_thread[id],
                    file_offset: records_per_thread.iter().take(id).sum::<usize>(),
                };

                let handle = s.spawn(move || glove_thread(&task_spec, &model_state, config));
                handles.push(handle);
            }
            for (i, handle) in handles.into_iter().enumerate() {
                match handle.join() {
                    Ok(Ok(cost)) => {
                        if config.verbose > 2 {
                            eprintln!("Thread {i} finished successfully.");
                        }
                        final_cost += cost;
                    }
                    Ok(Err(e)) => {
                        panic!("Thread {i} failed with an I/O error: {e}");
                    }
                    Err(_panic_payload) => {
                        panic!("Thread {i} panicked! This is a bug.");
                    }
                }
            }
        });

        let time_str = Local::now().format("%x - %I:%M.%S%p");
        eprintln!(
            "{time_str}, iter: {it:03}, cost: {cost}",
            it = b + 1,
            cost = final_cost / num_records as f64
        );
    }

    save_params(&model.w.0, config, vocab_size, -1)?;

    Ok(())
}

/// Worker thread: Hogwild AdaGrad
fn glove_thread(task: &ThreadTask, model: &SharedModel, config: &Config) -> io::Result<f64> {
    let w_ptr = model.w.0.as_ptr() as *mut f64;
    let gradsq_ptr = model.gradsq.0.as_ptr() as *mut f64;

    let mut thread_cost = 0.0;
    let mut fin = File::open(&config.input_file)?;

    let start_offset = task.file_offset * mem::size_of::<Crec>();
    fin.seek(SeekFrom::Start(start_offset as u64))?;

    let mut w_updates1 = vec![0.0f64; config.vector_size];
    let mut w_updates2 = vec![0.0f64; config.vector_size];
    let mut reader = BufReader::with_capacity(8192, fin);
    for _ in 0..task.records_to_process {
        let Some(cr) = Crec::read_from_raw(&mut reader)? else {
            break;
        };

        // should never happen - unless corrupt value read or wrong input file passed
        debug_assert!(cr.word1 >= 1 && cr.word1 as usize <= model.vocab_size);
        debug_assert!(cr.word2 >= 1 && cr.word2 as usize <= model.vocab_size);
        if cr.word1 < 1 || cr.word2 < 1 {
            eprintln!("Skipping corrupt Crec: {}, {}", cr.word1, cr.word2);
            continue;
        }
        // l1: start of word1 weights
        // l2: start of word2 context weights
        // vector_size +1 for the bias term
        // vocab_size: offset for context vectors
        let l1 = (cr.word1 as usize - 1) * (config.vector_size + 1);
        let l2 = (cr.word2 as usize - 1 + model.vocab_size) * (config.vector_size + 1);

        // cost function: J = sum_i,j f(Xij)(word_i dot word_j + b_i + b_j - log(X_ij))^2  (eq 8, p.4)
        // f(x) = (x/x_max)^alpha if x<x_max, 1 otherwise
        unsafe {
            let diff = (0..config.vector_size)
                .map(|i| *w_ptr.add(i + l1) * *w_ptr.add(i + l2))
                .sum::<f64>()      // dot word1, word2
                + *w_ptr.add(config.vector_size + l1) // bias of word 1
                + *w_ptr.add(config.vector_size + l2) // bias of word 2
                - cr.val.ln(); // log of co-occurence

            // weighted error
            let fdiff = if cr.val > config.x_max {
                diff
            } else {
                (cr.val / config.x_max).powf(config.alpha) * diff
            };

            if diff.is_nan() || fdiff.is_nan() {
                continue;
            }

            // Derivative of J = 0.5 * g(x)^2 = g(x) * g'(x)
            thread_cost += 0.5 * fdiff * diff;

            for i in 0..config.vector_size {
                let grad1 = fdiff * *w_ptr.add(i + l2); // gradient for w_i
                let grad2 = fdiff * *w_ptr.add(i + l1); // gradient for w_j
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

    Ok(thread_cost)
}

fn initialize_parameters(config: &Config, vocab_size: usize) -> GloveModel {
    let seed = if config.seed == 0 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    } else {
        config.seed
    };
    eprintln!("Using random seed {seed}");
    let mut rng = StdRng::seed_from_u64(seed);
    let w_size = 2 * vocab_size * (config.vector_size + 1);
    let mut w_vec = vec![0.0f64; w_size];
    for val in w_vec.iter_mut() {
        *val = (rng.random::<f64>() - 0.5) / config.vector_size as f64;
    }
    let gradsq_vec = vec![1.0f64; w_size];
    GloveModel {
        w: Arc::new(UnsafeSyncVec(w_vec)),
        gradsq: Arc::new(UnsafeSyncVec(gradsq_vec)),
    }
}

fn save_params(w: &[f64], config: &Config, vocab_size: usize, iter: i32) -> io::Result<()> {
    let save_file_str = config.save_file.to_str().unwrap();

    if config.use_binary > 0 {
        let bin_filename = if iter < 0 {
            format!("{save_file_str}.bin")
        } else {
            format!("{save_file_str}.{iter:03}.bin")
        };
        eprintln!("Saving final parameters to {bin_filename}");
        let mut f_out = BufWriter::new(File::create(bin_filename)?);
        f_out.write_all(bytemuck::cast_slice(w))?;
    }

    if config.use_binary != 1 {
        let txt_filename = if iter < 0 {
            format!("{save_file_str}.txt")
        } else {
            format!("{save_file_str}.{iter:03}.txt")
        };
        eprintln!("Saving final parameters to {txt_filename}");
        let mut f_out = BufWriter::new(File::create(txt_filename)?);
        let vocab_file = BufReader::new(File::open(&config.vocab_file)?);

        let vocab: Vec<String> = vocab_file
            .lines()
            .map(|line| line.unwrap().split_whitespace().next().unwrap().to_string())
            .collect();

        for (i, word) in vocab.iter().enumerate().take(vocab_size) {
            write!(f_out, "{word}")?;
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

fn calculate_records_per_thread(num_records: usize, num_threads: usize) -> Vec<usize> {
    if num_threads == 0 {
        return vec![];
    }
    let base = num_records / num_threads;
    let extra = num_records % num_threads;
    (0..num_threads)
        .map(|i| if i < extra { base + 1 } else { base })
        .collect()
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
        grad_clip_value: 100.0,
        verbose: cli.verbose,
        seed: cli.seed,
    };

    train_glove(&config)
}
