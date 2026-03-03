# VirusNet_subsampled_sim

Evaluation dataset containing both real subsampled genome fragments and
synthetic simulated sequences. Used to test model robustness on different
types of input.

## Source

Downloaded from:
`https://research.bifo.helmholtz-hzi.de/downloads/genomenet/VirusNet_subsampled_sim.tar.gz`

SHA-256: `7b9f936b34fc654b0b9964cb15709abf8a09df412a86272075a42d424922da12`

## Structure

```
VirusNet_subsampled_sim/
└── test/     ~8,069+ FASTA files
```

Note: The archive may contain additional train/validation splits deeper in the
file (only the first ~200 MB was inspected; total archive is 6.1 GB).

## File Types

### Real genome assemblies (`GCA_*_genomic.fasta`)

- NCBI GenBank assembly accessions (e.g. `GCA_900460085.1_41906_E02_genomic.fasta`)
- Subsampled fragments from real microbial genome assemblies
- Represent actual biological sequences

### Simulated sequences (`GCA_*_genomic_sim.fasta`)

- Paired with real assemblies (same accession, `_sim` suffix)
- Synthetically generated to match the nucleotide composition of the
  corresponding real genome
- Allow direct comparison of model performance on real vs. simulated input

### PLSDB plasmids (`PLSDB*.fasta`)

- Plasmid sequences from the Plasmid Database
- ~4,534 entries (majority of files in the test set)
- No corresponding `_sim` variants

## Composition (from first 200 MB)

| Type | Count |
|------|-------|
| Simulated GCA sequences | ~1,763 |
| Real GCA sequences | variable |
| PLSDB plasmids | ~4,534 |

## Role in Training

Primarily an **evaluation/test** dataset. The paired real+simulated design
allows measuring whether the model generalizes beyond real sequences to
synthetic ones with similar composition.
