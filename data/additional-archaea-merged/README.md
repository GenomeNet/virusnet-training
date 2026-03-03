# additional-archaea-merged

Archaea genome sequences used as additional non-virus training data to reduce
false positives on archaeal input.

## Source

Downloaded from:
`https://research.bifo.helmholtz-hzi.de/downloads/genomenet/additional-archaea-merged.tar.gz`

SHA-256: `ca51731565c5e8c6167fb197bf875fc95c5de27148c42535099db5f68f8c51e3`

## Structure

```
additional-archaea-merged/
├── train/       1,933 files (1,508 subsampled + 425 simulated)
├── validation/    610 files (  516 subsampled +  94 simulated)
└── test/          560 files (  441 subsampled + 119 simulated)
```

Total: 3,103 FASTA files.

## File Types

### Subsampled fragments (`subsample_*.fasta`)

- Each file contains **one sequence** of exactly **4,000 bp**
- Subsampled (randomly extracted) from real archaea genome assemblies
- Header format: `>subsample_<id>` (numeric ID, 0–1507)
- No species/accession info preserved in headers

### Simulated genomes (`simulated_sequence_*.fasta`)

- Each file contains **one sequence** of ~**870,000–890,000 bp**
- Synthetically generated sequences with archaeal nucleotide composition
- Header format: `>simulated_sequence_<id>` (numeric ID, 1–425)
- Generated to augment training data with sequences that have similar
  statistical properties to archaea but are not real genomes

## Role in Training

Used as non-virus (negative class) training data, specifically to teach the
model that archaea are not viruses. Without this dataset, the model may
misclassify archaeal sequences as viral due to some shared genomic features.
