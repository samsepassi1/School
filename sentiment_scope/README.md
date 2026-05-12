# SentimentScope — IMDB Sentiment Analysis (CineScope project)

A small GPT-style transformer (`DemoGPT`) trained **from scratch** on the IMDB
movie-review dataset to classify reviews as positive or negative.

## Layout

```
sentiment_scope/
├── SentimentScope.ipynb   main submission notebook (load -> explore -> dataset -> model -> train -> test)
├── build_notebook.py      regenerates SentimentScope.ipynb from Python source
├── inference.py           standalone CLI: predict sentiment for one or more reviews
├── requirements.txt
└── README.md              this file
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook SentimentScope.ipynb
```

The notebook runs end-to-end on a single GPU (Colab T4 / similar) in roughly
8–12 minutes for 4 epochs at `batch_size=32`, `max_len=256`. CPU works but is
much slower.

## Rubric mapping

| Rubric criterion | Where it's covered |
|---|---|
| Load train/test via helper + shape asserts | `load_imdb()` + `assert train_df.shape == (25000, 2)` |
| Descriptive stats + ≥ 2 labelled visualisations | `value_counts()` + label-balance and review-length plots |
| Custom `IMDBDataset` (`__init__/__len__/__getitem__`) with shape asserts | `IMDBDataset` cell |
| `DemoGPT.__init__` / `forward` for binary classification + dummy-input shape assert | `DemoGPT` cell (`out.shape == (4, 2)`) |
| `calculate_accuracy()` helper | `calculate_accuracy` cell |
| Training loop runs without errors | `train_model()` + driver cell |
| Test accuracy > 75% | final eval cell (`assert test_acc > 0.75`) |
| Project report + ≥ 2 takeaways | "Report — Results & Key Takeaways" markdown cell |
| Stand-out: load checkpoint and run inference on a batch | `SentimentScopeInference` class + `inference.py` |

## Results

Typical run (4 epochs, AdamW + OneCycle LR, dropout 0.1):

| split | accuracy |
|---|---|
| validation (best) | ~0.87 |
| **test (25,000)** | **~0.86** |

Checkpoint is saved to `demogpt_imdb.pt` during training and loaded back for
test-time evaluation and inference.

## Standalone inference

After training, you can predict sentiment from the command line:

```bash
python inference.py --ckpt demogpt_imdb.pt \
    "Absolutely loved this film." \
    "Wooden acting, non-existent plot."
```
