# Training Data

This directory holds the training datasets for VirusNet binary classification.
Data files (`.fasta`, `.tar.gz`) are not checked into git — download them with:

```bash
python download.py            # download everything
python download.py --list     # show available assets
```

## Directory Layout

After downloading and extraction:

```
data/
├── VirusNet_data_virus/        # Positive class (virus sequences)
│   ├── train/                  #     746 FASTAs (369 GenBank + 377 MGV vOTU)
│   ├── validation/             #     213 FASTAs (102 GenBank + 111 MGV vOTU)
│   ├── test/                   #     107 FASTAs ( 53 GenBank +  54 MGV vOTU)
│   └── train_small/            #       3 FASTAs (sanity-check subset)
├── non_virus/                  # Primary negative class
│   ├── train/                  #   20,707 PLSDB plasmids + 381 VGP assemblies
│   └── test/                   #    6,903 PLSDB plasmids +   8 VGP assemblies
├── additional-archaea-merged/  # Archaea (non-virus augmentation)
│   ├── train/                  #   1,508 subsample + 425 simulated
│   ├── validation/             #     516 subsample +  94 simulated
│   └── test/                   #     441 subsample + 119 simulated
├── VirusNet_subsampled_sim/    # Large-scale subsampled + simulated
│   ├── train/                  # 160,690 files
│   ├── validation/             #  51,338 files
│   └── test/                   #  49,552 files
└── bio_bakery/                 # BioBakery microbial genomes (optional, 125 GB)
    └── test/
```

Each subdirectory has its own `README.md` with detailed provenance information.

## Dataset Summary

| Dataset | Files | Positive/Negative | Key Contents |
|---------|-------|-------------------|--------------|
| **VirusNet_data_virus** | 1,069 | Positive (virus) | NCBI GenBank viral genomes + MGV gut-virome vOTU representatives (~100 records per file) |
| **non_virus** | 27,999 | Negative | PLSDB plasmids (11,206 organisms, 30+ genera) + VGP eukaryotic assemblies (4 species) |
| **additional-archaea-merged** | 3,103 | Negative | 4 kb archaea subsamples + ~1 Mb simulated sequences |
| **VirusNet_subsampled_sim** | 261,580 | Negative (non-virus) | 99,132 NCBI assemblies (40 kb real + 100 kb sim) + 62,274 PLSDB plasmids |
| **bio_bakery** | TBD | Negative | Large BioBakery microbial reference genomes |

## Train/Test Split Strategies

- **non_virus**: PLSDB plasmids split by accession (no filename overlap).
  VGP assemblies split by chromosome (train: chr 1–7, test: chr 8–10).
- **additional-archaea-merged**: Independent ID numbering per split (no leakage).
- **VirusNet_subsampled_sim**: Split by GCA accession across train/validation/test.

## Taxonomic Coverage (non-virus negative class)

The non-virus datasets collectively cover:
- **Bacteria**: 11,206 unique organisms from PLSDB (top genera: *Escherichia*,
  *Klebsiella*, *Enterococcus*, *Salmonella*, *Staphylococcus*, *Acinetobacter*,
  *Bacillus*, *Enterobacter*, and many more)
- **Archaea**: Unidentified species (headers anonymized in subsampled fragments)
- **Eukaryotes**: *Homo sapiens*, *Microcaecilia unicolor*, *Acanthisitta chloris*,
  *Alopias superciliosus* (from VGP)
- **Microbial diversity**: 99,132 NCBI genome assemblies in VirusNet_subsampled_sim
- **BioBakery**: Additional broad microbial reference diversity (125 GB)
