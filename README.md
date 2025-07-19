# GloVe-rs

Rusty GloVe - Rust implementation of [GloVe](https://github.com/stanfordnlp/GloVe).

## References

1. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
2. ["GloVe: Global Vectors for Word Representation", Jeffrey Pennington,   Richard Socher,   Christopher D. Manning](https://nlp.stanford.edu/pubs/glove.pdf)

## Run

Download text8 (Wikipedia, 100M characters, 17.5M words):

```
% wget http://mattmahoney.net/dc/text8.zip
% unzip text8.zip
% wc text8

0 17005207 100000000 text8
```

Train and evaluate
```
% time sh demo.sh

SEMANTIC ANALOGY TESTS
File: capital-common-countries.txt — Accuracy: 62.06% (314/506)
File: capital-world.txt — Accuracy: 24.83% (885/3564)
File: currency.txt — Accuracy: 4.36% (26/596)
File: city-in-state.txt — Accuracy: 26.09% (608/2330)
File: family.txt — Accuracy: 38.81% (163/420)
SEMANTIC Total Accuracy: 26.91% (1996/7416)
SEMANTIC Questions seen/total: 83.62% (7416/7416)

SYNTACTIC ANALOGY TESTS
File: gram1-adjective-to-adverb.txt — Accuracy: 5.24% (52/992)
File: gram2-opposite.txt — Accuracy: 3.44% (26/756)
File: gram3-comparative.txt — Accuracy: 27.55% (367/1332)
File: gram4-superlative.txt — Accuracy: 5.34% (53/992)
File: gram5-present-participle.txt — Accuracy: 11.84% (125/1056)
File: gram6-nationality-adjective.txt — Accuracy: 57.26% (871/1521)
File: gram7-past-tense.txt — Accuracy: 15.26% (238/1560)
File: gram8-plural.txt — Accuracy: 26.20% (349/1332)
File: gram9-plural-verbs.txt — Accuracy: 6.67% (58/870)
SYNTACTIC Total Accuracy: 20.55% (2139/10411)
SYNTACTIC Questions seen/total: 97.53% (10411/10411)

OVERALL RESULTS:
Total Accuracy: 23.20% (4135/17827)
Total Questions seen/total: 91.21% (17827/19544)

sh demo.sh  698.23s user 27.69s system 566% cpu 2:08.21 total
```

Word analogy - Interactive
```
% cargo run --bin word_analogy --release

Word analogy - KING is to QUEEN as MAN is to ?
Enter 3 words: king queen man
  1:  0.90054 woman
  2:  0.82080 beautiful
  3:  0.79237 girl
  4:  0.77680 lady
  5:  0.77009 my
  6:  0.72895 she
  7:  0.72797 animal
  8:  0.70304 child
  9:  0.69457 bird
 10:  0.69252 baby
 11:  0.68946 her
 12:  0.68284 mother
 13:  0.67897 eyes
 14:  0.67114 person
 15:  0.67079 horse
 16:  0.66328 young
 17:  0.66307 dog
 18:  0.66300 love
 19:  0.66263 a
 20:  0.66069 thing
 21:  0.65655 every
 22:  0.65592 whole
 23:  0.65369 your
 24:  0.64499 princess
 25:  0.64326 husband
 26:  0.64320 face
 27:  0.64040 female
 28:  0.63627 soul
 29:  0.63602 eating
 30:  0.63438 men
```
