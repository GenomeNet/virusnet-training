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
│   ├── train/
│   └── validation/
├── non_virus/                  # Negative class (non-virus sequences)
│   ├── train/                  #   PLSDB plasmids, VGP assemblies
│   └── test/
├── additional-archaea-merged/  # Additional archaea (non-virus augmentation)
│   ├── train/
│   ├── validation/
│   └── test/
├── VirusNet_subsampled_sim/    # Evaluation data (real + simulated)
│   └── test/
└── bio_bakery/                 # BioBakery microbial genomes (optional, large)
    └── test/
```

Each subdirectory has its own `README.md` with detailed provenance information.
