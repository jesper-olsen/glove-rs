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
