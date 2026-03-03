# VirusNet_data_virus

Virus genome sequences — the **positive class** for binary classification.

## Source

Downloaded from:
`https://research.bifo.helmholtz-hzi.de/downloads/genomenet/VirusNet_data_virus.tar.gz`

SHA-256: not yet pinned

## Structure

```
VirusNet_data_virus/
├── train/        virus FASTA files for training
└── validation/   virus FASTA files for validation
```

## Role in Training

This is the positive (virus) class for the binary VirusNet classifier. Combined
with the non-virus datasets, it forms the complete training set for fine-tuning
the pre-trained BERT model to distinguish viral from non-viral DNA sequences.
