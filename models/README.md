# Models

Pre-trained model weights. Download with:

```bash
python download.py model
```

## llm_1k_bert.h5

Base BERT model pre-trained with masked language modeling (MLM) on virus genome
sequences using deepG. This is the starting point for fine-tuning the binary
VirusNet classifier.

- **SHA-256**: `27bddd035ba38d783373e6703f71ac5c812789a9a48f7c5783c0841d64dd52f4`
- **Size**: 2.8 GB
- **Input**: `(batch, 1000)` — 1-gram integer-encoded DNA sequences
- **Vocabulary**: 6 tokens (A=1, C=2, G=3, T=4, pad/N=0, MASK=5)
- **Architecture**: 18 transformer blocks, 20 heads, embed_dim=650, ff_dim=1950
- **Output (pre-training)**: Dense(256, relu) -> Dense(6, softmax)

See the main README for full architecture details.
