# VirusNet_data_virus

Virus genome sequences — the **positive class** for binary classification.

## Source

Downloaded from:
`https://research.bifo.helmholtz-hzi.de/downloads/genomenet/VirusNet_data_virus.tar.gz`

SHA-256: `61f671a7e332f4e77590541817cb45b2e963e43916be1bf2bfa4d4333455d9f3`

Snapshot taken from
`/vol/projects/BIFO/genomenet/training_data/VirusNet_data_virus_2023-07-20/`
(July 2023). Tarball is 1.08 GB compressed / ~1.7 GB extracted.

## Structure

```
VirusNet_data_virus/
├── train/         746 FASTA files
├── validation/    213 FASTA files
├── test/          107 FASTA files
└── train_small/     3 FASTA files (mini sanity-check subset)
```

## Contents

Two source corpora, both virus-only, batched ~100 records per file:

| Prefix | train | validation | test | Source |
|---|---:|---:|---:|---|
| `GenBank_genomes_*.fasta` | 369 | 102 | 53 | NCBI GenBank viral genomes/segments (RefSeq + GenBank accessions, e.g. `NC_025115.1 Ralstonia phage RSY1 …`) |
| `mgv_votu_representatives_*.fasta` | 377 | 111 | 54 | Metagenomic Gut Virus (MGV) catalog vOTU representatives (e.g. `MGV-GENOME-0011269 OTU-10020`) |

## Role in Training

Positive class for the binary VirusNet classifier. Combined with the non-virus
datasets (`non_virus`, `additional-archaea-merged`, `bio_bakery`,
`VirusNet_subsampled_sim`) it forms the complete training set for fine-tuning
the pre-trained BERT model to distinguish viral from non-viral DNA.
