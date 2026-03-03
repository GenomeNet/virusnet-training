# bio_bakery

Large collection of microbial reference genomes derived from the BioBakery
framework. Used as additional non-virus evaluation/training data providing
broad microbial diversity.

## Source

Downloaded from:
`https://research.bifo.helmholtz-hzi.de/downloads/genomenet/bio_bakery.tar.gz`

SHA-256: not yet pinned (125 GB file)

## Structure

```
bio_bakery/
└── test/     large number of FASTA files
```

Note: Due to the archive size (125 GB), only the first few MB were inspected.
Additional train/validation splits may exist deeper in the archive.

## File Types

### NCBI genome assemblies (`GCA_*_genomic.fasta`)

- NCBI GenBank assembly accessions
- Microbial genome sequences from the BioBakery reference database
- Examples: `GCA_900460085.1_41906_E02_genomic.fasta`

## BioBakery Background

BioBakery (https://huttenhower.sph.harvard.edu/biobakery/) is a collection of
tools and reference databases for microbial community profiling. The reference
genomes included here likely originate from the MetaPhlAn/ChocoPhlAn marker
gene databases or the HUMAnN reference pangenome collection, representing a
comprehensive set of known microbial species.

## Role in Training

Provides **broad microbial diversity** beyond what PLSDB plasmids and the
handful of VGP assemblies in `non_virus/` cover. This is a large optional
dataset (125 GB); training can proceed without it, but including it may improve
the model's ability to correctly classify diverse microbial sequences as
non-viral.
