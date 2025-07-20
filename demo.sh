cargo run --release --bin vocab -- --min-count 5 DATA/text8 >vocab.txt
cargo run --release --bin cooccur -- --vocab-file vocab.txt < DATA/text8 > cooccur.bin
cargo run --release --bin shuffle_clap < cooccur.bin >cooccur_shuffled.bin
cargo run --release --bin shuffle < cooccur.bin >cooccur_shuffled.bin
cargo run --release --bin glove -- --input-file cooccur_shuffled.bin --vocab-file vocab.txt --save-file vectors
cargo run --release --bin eval 
