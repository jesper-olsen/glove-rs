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
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
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

/// Holds the model parameters (word vectors and their squared gradients).
struct GloveModel {
    w: Arc<Vec<AtomicU64>>,
    gradsq: Arc<Vec<AtomicU64>>,
}

/// Contains references to the shared model data.
struct SharedModel<'a> {
    w: &'a Arc<Vec<AtomicU64>>,
    gradsq: &'a Arc<Vec<AtomicU64>>,
    vocab_size: usize,
}

/// Defines the specific workload for a single thread.
struct ThreadTask {
    records_to_process: usize,
    file_offset: usize,
}

fn train_model_parallel<F>(
    config: &Config,
    vocab_size: usize,
    num_records: usize,
    model: &GloveModel,
    thread_fn: F,
) -> Result<(), Box<dyn Error>>
where
    F: Fn(&ThreadTask, &SharedModel, &Config) -> io::Result<f64> + Send + Sync + Copy,
{
    let records_per_thread = calculate_records_per_thread(num_records, config.num_threads);
    let mut file_offsets = vec![0; config.num_threads];
    for i in 1..config.num_threads {
        file_offsets[i] = file_offsets[i - 1] + records_per_thread[i - 1];
    }

    for b in 0..config.num_iter {
        let final_cost = thread::scope(|s| {
            let handles = (0..config.num_threads)
                .map(|id| {
                    let model_state = SharedModel {
                        w: &model.w,
                        gradsq: &model.gradsq,
                        vocab_size,
                    };
                    let task_spec = ThreadTask {
                        records_to_process: records_per_thread[id],
                        file_offset: file_offsets[id],
                    };
                    s.spawn(move || thread_fn(&task_spec, &model_state, config))
                })
                .collect::<Vec<_>>();

            handles
                .into_iter()
                .enumerate()
                .map(|(i, handle)| match handle.join() {
                    Ok(Ok(cost)) => {
                        if config.verbose > 2 {
                            eprintln!("Thread {i} finished successfully.");
                        }
                        cost
                    }
                    Ok(Err(e)) => {
                        panic!("Thread {i} failed with an I/O error: {e}");
                    }
                    Err(_) => {
                        panic!("Thread {i} panicked!");
                    }
                })
                .sum::<f64>()
        });

        let time_str = Local::now().format("%x - %I:%M.%S%p");
        eprintln!(
            "{time_str}, iter: {it:03}, cost: {cost}",
            it = b + 1,
            cost = final_cost / num_records as f64
        );
    }

    Ok(())
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

    train_model_parallel(config, vocab_size, num_records, &model, glove_thread)?;

    save_params(&model.w, config, vocab_size, -1)?;
    Ok(())
}

/// Worker thread: Atomic AdaGrad
fn glove_thread(task: &ThreadTask, model: &SharedModel, config: &Config) -> io::Result<f64> {
    let mut thread_cost = 0.0;
    let mut fin = File::open(&config.input_file)?;

    let start_offset = task.file_offset * mem::size_of::<Crec>();
    fin.seek(SeekFrom::Start(start_offset as u64))?;

    let mut reader = BufReader::with_capacity(8192, fin);

    // All shared memory access is done through safe atomic methods.
    // Using Relaxed ordering is sufficient and fastest. It guarantees atomicity for
    // individual operations but doesn't add stronger synchronization overhead,
    // which mimics the Hogwild! spirit while being memory-safe.
    let ordering = Ordering::Relaxed;

    // short-cuts - help compiler see through pointer indirections
    let w_slice: &[AtomicU64] = &model.w[..];
    let gradsq_slice: &[AtomicU64] = &model.gradsq[..];

    for _ in 0..task.records_to_process {
        let Some(cr) = Crec::read_from_raw(&mut reader)? else {
            break;
        };

        let l1 = (cr.word1 as usize) * (config.vector_size + 1);
        let l2 = (cr.word2 as usize + model.vocab_size) * (config.vector_size + 1);

        // Racy, but that's the point of Hogwild!
        let dot_product: f64 = (0..config.vector_size)
            .map(|i| {
                let w1_bits = w_slice[i + l1].load(ordering);
                let w2_bits = w_slice[i + l2].load(ordering);
                f64::from_bits(w1_bits) * f64::from_bits(w2_bits)
            })
            .sum();

        let w_bias1_bits = w_slice[config.vector_size + l1].load(ordering);
        let w_bias2_bits = w_slice[config.vector_size + l2].load(ordering);
        let diff =
            dot_product + f64::from_bits(w_bias1_bits) + f64::from_bits(w_bias2_bits) - cr.val.ln();

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
            let w_l1_bits = w_slice[i + l1].load(ordering);
            let w_l2_bits = w_slice[i + l2].load(ordering);
            let gradsq_l1_bits = gradsq_slice[i + l1].load(ordering);
            let gradsq_l2_bits = gradsq_slice[i + l2].load(ordering);

            let w_l1 = f64::from_bits(w_l1_bits);
            let w_l2 = f64::from_bits(w_l2_bits);
            let gradsq_l1 = f64::from_bits(gradsq_l1_bits);
            let gradsq_l2 = f64::from_bits(gradsq_l2_bits);

            let grad1 = fdiff * w_l2;
            let grad2 = fdiff * w_l1;

            let temp1 = grad1.clamp(-config.grad_clip_value, config.grad_clip_value) * config.eta;
            let temp2 = grad2.clamp(-config.grad_clip_value, config.grad_clip_value) * config.eta;

            // Separate load/calculate/store - fetch_add would syncronise
            w_slice[i + l1].store((w_l1 - temp1 / gradsq_l1.sqrt()).to_bits(), ordering);
            w_slice[i + l2].store((w_l2 - temp2 / gradsq_l2.sqrt()).to_bits(), ordering);

            gradsq_slice[i + l1].store((gradsq_l1 + temp1 * temp1).to_bits(), ordering);
            gradsq_slice[i + l2].store((gradsq_l2 + temp2 * temp2).to_bits(), ordering);
        }

        // Same load/calculate/store pattern for the biases
        let w_bias1_bits = w_slice[config.vector_size + l1].load(ordering);
        let w_bias2_bits = w_slice[config.vector_size + l2].load(ordering);
        let gradsq_b1_bits = gradsq_slice[config.vector_size + l1].load(ordering);
        let gradsq_b2_bits = gradsq_slice[config.vector_size + l2].load(ordering);

        let w_b1 = f64::from_bits(w_bias1_bits);
        let w_b2 = f64::from_bits(w_bias2_bits);
        let gradsq_b1 = f64::from_bits(gradsq_b1_bits);
        let gradsq_b2 = f64::from_bits(gradsq_b2_bits);

        let bias_grad = fdiff * config.eta;
        let update1 = bias_grad / gradsq_b1.sqrt();
        let update2 = bias_grad / gradsq_b2.sqrt();

        w_slice[config.vector_size + l1].store((w_b1 - check_nan(update1)).to_bits(), ordering);
        w_slice[config.vector_size + l2].store((w_b2 - check_nan(update2)).to_bits(), ordering);

        let fdiff_sq = fdiff * fdiff;
        gradsq_slice[config.vector_size + l1].store((gradsq_b1 + fdiff_sq).to_bits(), ordering);
        gradsq_slice[config.vector_size + l2].store((gradsq_b2 + fdiff_sq).to_bits(), ordering);
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

    // Create the initial weight vectors as f64
    let mut w_vec_f64 = vec![0.0f64; w_size];
    for val in w_vec_f64.iter_mut() {
        *val = (rng.random::<f64>() - 0.5) / config.vector_size as f64;
    }
    let w_vec = w_vec_f64
        .into_iter()
        .map(|f_val| AtomicU64::new(f_val.to_bits()))
        .collect();

    // Create the initial gradient squared vectors as f64
    let gradsq_vec_f64 = vec![1.0f64; w_size];
    let gradsq_vec = gradsq_vec_f64
        .into_iter()
        .map(|f_val| AtomicU64::new(f_val.to_bits()))
        .collect();

    GloveModel {
        w: Arc::new(w_vec),
        gradsq: Arc::new(gradsq_vec),
    }
}

fn save_params(
    w_atomic: &[AtomicU64],
    config: &Config,
    vocab_size: usize,
    iter: i32,
) -> io::Result<()> {
    // Atomically load all values into a plain Vec<f64> for saving.
    let w: Vec<f64> = w_atomic
        .iter()
        .map(|atomic_u64| {
            let bits = atomic_u64.load(Ordering::Relaxed);
            f64::from_bits(bits)
        })
        .collect();

    let save_file_str = config.save_file.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Save file path is not valid UTF-8",
        )
    })?;

    if config.use_binary > 0 {
        let bin_filename = if iter < 0 {
            format!("{save_file_str}.bin")
        } else {
            format!("{save_file_str}.{iter:03}.bin")
        };
        eprintln!("Saving final parameters to {bin_filename}");
        let mut f_out = BufWriter::new(File::create(bin_filename)?);
        f_out.write_all(bytemuck::cast_slice(&w))?;
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
