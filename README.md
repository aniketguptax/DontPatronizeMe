# 70016 NLP Coursework: PCL Detection (BestModel)

This repository contains my implementation for “Don’t Patronise Me!”, with the best run saved under `BestModel`.

If reviewing quickly, start here:
- Main notebook: [`BestModel/notebook.ipynb`](BestModel/notebook.ipynb)
- Best model weights: [`BestModel/best_multitask_model.pt`](BestModel/best_multitask_model.pt) (Git LFS)
- Prediction files: [`dev.txt`](dev.txt), [`test.txt`](test.txt)
- Final test CSV (for traceability): [`test_predictions.csv`](test_predictions.csv)

---

## What this project does

Task: binary Patronising and Condescending Language detection (PCL vs non-PCL), where:
- `PCL = 1` if original label `>= 2`
- `non-PCL = 0` otherwise

Model approach used for the submitted run:
- Shared encoder (`microsoft/deberta-v3-base`, fallback to `roberta-base`)
- Two heads:
  - binary head (`1` logit)
  - auxiliary taxonomy head (`7` logits, multi-label)
- Training objective:
  - `BCEWithLogitsLoss` for binary task with positive-class weighting
  - `BCEWithLogitsLoss` for taxonomy task
  - combined as `loss_bin + lambda_tax * loss_tax`
- Threshold tuned on dev set to maximise positive-class F1

---

## Data and links

### Data files
- Full dataset: [`data/dontpatronizeme_pcl.tsv`](data/dontpatronizeme_pcl.tsv)
- Train taxonomy split: [`data/train_data.csv`](data/train_data.csv)
- Dev taxonomy split: [`data/dev_data.csv`](data/dev_data.csv)
- Official test input (no labels): [`data/test_data.tsv`](data/test_data.tsv)

### External references
- SemEval 2022 Task 4 repository (Don’t Patronise Me): <https://github.com/PerezAlmendrosC/dontpatronizeme>
- SemEval 2022 Task 4 paper (ACL Anthology): <https://aclanthology.org/2022.semeval-1.38/>
- DeBERTa-v3-base model card: <https://huggingface.co/microsoft/deberta-v3-base>
- RoBERTa-base model card: <https://huggingface.co/roberta-base>

---

## Environment and dependencies

Python 3.10+ recommended.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:
- pandas
- numpy
- matplotlib
- scikit-learn
- torch
- transformers
- tqdm
- sentencepiece

Optionallu:
- Set `HF_TOKEN` for faster Hugging Face downloads and higher rate limits.

---

## Reproducing the work

All implementation code for is inside the notebook (no separate `src/` modules are needed for this final version).

Open and run:
- [`BestModel/notebook.ipynb`](BestModel/notebook.ipynb)

What each stage does:
- Sweep: randomised subset of hyperparameter combinations on an internal train/dev split from the official train set
- Refinement: retrains top sweep candidates on official train/dev with longer schedule
- Final run: trains once with best refined settings, tunes threshold on official dev, then writes final dev/test predictions

---

## Final selected configuration

From refinement, the selected configuration was:
- `batch_size = 16`
- `max_length = 192`
- `learning_rate = 3e-5`
- `weight_decay = 0.01`
- `warmup_ratio = 0.10`
- `gradient_clip_norm = 0.8`
- `lambda_tax = 0.5`
- `num_epochs = 8`
- `early_stopping_patience = 3`

Best tuned threshold on dev:
- `tau = 0.21`

---

## Final results from the submitted run

Official dev set:
- At `tau = 0.50`:
  - Precision: `0.4675`
  - Recall: `0.5779`
  - F1: `0.5169`
- At tuned `tau = 0.21`:
  - Precision: `0.4437`
  - Recall: `0.6332`
  - F1: `0.5217`

Confusion matrix at tuned threshold:
- `TN = 1737`
- `FP = 158`
- `FN = 73`
- `TP = 126`

---

## Notes about the model file size (Git LFS)

The model checkpoint is large, so it is tracked with Git LFS:
- [`BestModel/best_multitask_model.pt`](BestModel/best_multitask_model.pt)

After cloning, fetch LFS objects with:

```bash
git lfs pull
```
