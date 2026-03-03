# non_virus

Diverse collection of non-viral genomic sequences used as the primary negative
class for binary classification (virus vs. non-virus).

## Source

Downloaded from:
`https://research.bifo.helmholtz-hzi.de/downloads/genomenet/non_virus.tar.gz`

SHA-256: `f403f54034edbc36278a051cd6b0d97ba5f8e859149fa48a388157b48401c11b`

## Structure

```
non_virus/
├── train/        ~20,239 FASTA files
├── test/          ~6,911 FASTA files
├── train_sim/     (empty, placeholder for simulated sequences)
├── test_sim/      (empty, placeholder for simulated sequences)
└── run.sh         simulation script (see below)
```

Total: ~27,150 FASTA files.

## File Types and Sources

### PLSDB plasmids (`PLSDB*.fasta`)

- From the Plasmid Database (https://ccb-microbe.cs.uni-saarland.de/plsdb/)
- Header format: `>NZ_CP... <organism> plasmid <name>, complete sequence`
- Represent bacterial plasmid sequences (circular, variable length)
- Bulk of the dataset

### VGP/Genome Ark assemblies (`<species_code>.pri.cur.*.fasta`)

Eukaryotic genome fragments from the Vertebrate Genomes Project and related
assemblies:

| Prefix | Species | Common Name |
|--------|---------|-------------|
| `aMicUni` | *Microcaecilia unicolor* | Tiny Cayenne caecilian (amphibian) |
| `bAcaChl` | *Acanthisitta chloris* | Rifleman (bird/bacteria) |
| `fAloSap` | *Alopias superciliosus* | Bigeye thresher shark (fish) |
| `mHomSap` | *Homo sapiens* | Human |

- Header format varies by assembly
- Fragments from primary curated assemblies (`.pri.cur.`)

### Sequence Properties

- Variable sequence lengths (2 kbp to ~100 kbp observed)
- One sequence per FASTA file (typical)

## Simulation Script (`run.sh`)

The included `run.sh` references `simple_sequence_simulator/process.py` with
parameters:

```bash
python process.py \
  --fragment_length 4000 \
  --n_fragments 500 \
  --input_dir train \
  --output_dir train_subsampled \
  --sim_output_dir train_sim \
  --sim_size_kb 100
```

This suggests the data was intended to be further processed:
- Fragment length: 4,000 bp (matching the archaea subsamples)
- 500 fragments per genome
- Simulated output at 100 kb

The `train_sim/` and `test_sim/` directories are empty, indicating this
simulation step was not yet completed for this dataset.

## Role in Training

Primary negative class. The diversity of sources (plasmids, eukaryotic genomes,
bacterial genomes) ensures the model learns to distinguish viruses from a broad
range of non-viral DNA.
