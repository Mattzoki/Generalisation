# Generalisation: NER Robustness Across Datasets

Code for my MA thesis, *Evaluating Generalisation in Named Entity Recognition through Robustness Testing* (Vrije Universiteit Amsterdam, 2025). NER models often score well on the dataset they were trained on and then drop on text from a different source. This project fine-tunes BERT on one NER dataset and measures how it holds up on another, on subsets of it, and on challenge sets.

The thesis PDF is in [Main-Papers](https://github.com/Mattzoki/Main-Papers).

## What it does

- Fine-tunes `bert-base-cased` and `bert-base-uncased` for token classification, trained on **CoNLL-2003** and on **OntoNotes** (labels mapped to PER, ORG, LOC and MISC in BIO format).
- Evaluates each model:
  - in-dataset (CoNLL→CoNLL, OntoNotes→OntoNotes)
  - cross-dataset (CoNLL→OntoNotes and back), with OntoNotes split into newswire and the rest
  - on challenge sets (a Stanford challenge set, a CoNLL-2003 challenge set from the Guided-Adversarial-Augmentation project, and a common-nouns challenge)
- Repeats training over several random seeds to check how stable the results are.
- Analyses errors per entity tag and lists misclassified tokens.

## Repository layout

| Path | Contents |
| --- | --- |
| `Matt.py` | Trains BERT on CoNLL-2003 and evaluates on OntoNotes over 5 seeds (42, 100, 2023, 7, 999), then plots F1 and accuracy per seed |
| `splits/` | Dataset splits saved with Hugging Face `datasets` (CoNLL, OntoNotes main/clean/newswire/rest, challenge sets) |
| `finals/` | Final experiment notebooks: `conll_*` and `onto_*` (train and evaluate, cased and uncased), `*_conll2conll` and `*_onto2onto` (in-dataset runs), `error_analysis.ipynb`, `visuals.ipynb` |
| `finals/*.csv` | Misclassified tokens and the common-nouns challenge set |
| `saved_model/` | Fine-tuned model checkpoints |
| `requirements.txt` | Conda environment export (Windows, Python 3.8, CUDA 11.8) |

## Setup

The environment was exported from conda on Windows, with PyTorch 2.4.1, Transformers 4.46.3 and CUDA 11.8:

```bash
conda create --name ner-gen --file requirements.txt
conda activate ner-gen
```

On other platforms, install `torch`, `transformers`, `datasets`, `evaluate`, `seqeval`, `scikit-learn`, `pandas`, `matplotlib` and `seaborn` yourself.

## Running

1. Build the split folders in `splits/` from CoNLL-2003 and OntoNotes. The repo keeps only their metadata, not the data files.
2. Train and evaluate with `python Matt.py`, which uses mixed precision (`fp16`), so a GPU is needed.
3. Open the notebooks in `finals/` to reproduce per-experiment results, error analysis and plots.

Training settings in `Matt.py`: learning rate 2e-5, batch size 8 with 2 gradient-accumulation steps, 3 epochs, weight decay 0.01.

## Notes

- The CoNLL-2003 challenge set is not included. `error_analysis.ipynb` expects it from the Guided-Adversarial-Augmentation project.
- Results are reported as seqeval F1 (entity-level) and per-tag F1.
