# Zero-Shot and Few-Shot Cross-Linguistic Transfer for Token-Level Hinglish Language Identification Using Multilingual Transformers

> Token-level language identification for Hindi-English (Hinglish) code-switched text using four multilingual transformer models benchmarked under identical training conditions.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Results](#results)
- [Linguistic Challenges](#linguistic-challenges)
- [Key Takeaways](#key-takeaways)
- [Authors](#authors)

---

## Overview

Code-switching — switching between two or more languages within a single conversation — is extremely common in multilingual societies like India. Hinglish (Hindi + English) is everywhere on social media and messaging platforms, but most NLP systems are built for monolingual input and fall apart on mixed-language text.

This project benchmarks four multilingual transformer models on **token-level language identification (LID)** for Hinglish. Each token in a sentence is labeled as one of: `lang1` (Hindi), `lang2` (English), `ne` (named entity), `other`, `fw` (foreign word), `mixed`, or `unk`.

Example:

| Token | Namaste | everyone | kal | meeting | hai | . |
|-------|---------|----------|-----|---------|-----|---|
| Label | lang1 | lang2 | lang1 | lang2 | lang1 | other |

---

## Motivation

Getting LID right at the token level is foundational for downstream tasks like POS tagging, named entity recognition, and sentiment analysis in code-switched settings. The problem is that:

- Annotated Hinglish corpora are small and hard to build
- Tokens like *is*, *to*, *me* exist in both Hindi and English — context is everything
- Hindi written in Roman script (*nahi / nai / naahi*) has no consistent spelling
- Hybrid words like *planning karna* or *confirm hua* don't exist in standard pretraining data

This work evaluates how well modern multilingual transformers handle all of this under **zero-shot and few-shot cross-linguistic transfer** — using pretrained multilingual knowledge to generalize to Hinglish with minimal supervision.

---

## Dataset

The dataset is a curated Hinglish token-tag corpus built from social media and informal text. Preprocessing included text normalization, punctuation isolation, Unicode cleanup, and label alignment for subword tokenizers.

**Label distribution:** `lang1` (Hindi) and `lang2` (English) dominate. Rare classes — `mixed`, `fw`, `unk` — have very low support, which directly impacts model performance on those tags.

**Sentence length breakdown:**
- Short (≤10 tokens): ~1,500 sentences
- Medium (11–20 tokens): ~1,200 sentences
- Long (>20 tokens): ~1,100 sentences

**Code-switch point distribution:**

| Switch Points | % of Sentences | What It Means |
|---|---|---|
| 0 | 21% | Monolingual segments |
| 1 | 48% | Single switch — most common |
| 2 | 19% | Two clear switch boundaries |
| 3 or more | 12% | Highly mixed, complex sentences |

Single-switch sentences dominate the dataset. Highly mixed sentences (3+ switches) are rare but the hardest for every model.

---

## Models

Four multilingual transformer models were evaluated, all fine-tuned with **identical hyperparameters** for a fair comparison:

| Model | Architecture Depth | Pretraining Scale | Inference Speed |
|---|---|---|---|
| `mDeBERTa-v3-base` | Highest | Large-scale multilingual | Moderate |
| `XLM-RoBERTa-base` | High | CC100 corpus (2.5TB, 100+ languages) | Moderate |
| `BERT-base-multilingual-cased` (mBERT) | Medium | Multilingual Wikipedia (104 languages) | Moderate |
| `DistilBERT-multilingual-cased` | Low | Distilled from mBERT | Fastest |

**Fine-tuning settings (same across all models):**
- Epochs: 3
- Learning rate: 2×10⁻⁵
- Optimizer: AdamW with linear warmup scheduler
- Mixed precision: fp16
- Evaluation strategy: per epoch
- Label alignment: first subtoken gets the gold label; remaining subtokens are masked with `-100`

---

## Methodology

1. **Data Preparation** — Raw Hinglish text normalized and tokenized into a HuggingFace `DatasetDict` with train/dev/test splits
2. **Label Alignment** — Each model's subword tokenizer maps original token labels to subword sequences without misalignment
3. **Model Setup** — All four models loaded via `AutoModelForTokenClassification` with a classification head on top of the encoder
4. **Fine-tuning** — HuggingFace `Trainer` API used with identical settings across all models
5. **Evaluation** — Per-class precision, recall, F1; micro/macro/weighted averages; overall accuracy; inference runtime
6. **Zero/Few-shot Transfer Analysis** — Error inspection on rare tags (`fw`, `mixed`, `unk`) and morphologically hybrid tokens

---

## Results

### Overall Performance

| Model | Accuracy | F1-Score |
|---|---|---|
| **mDeBERTa-v3-base** | **0.9668** | **0.8816** |
| XLM-RoBERTa-base | 0.9644 | 0.8750 |
| mBERT | 0.9532 | — |
| DistilBERT-multilingual-cased | 0.9410 | — |

**mDeBERTa-v3-base is the clear winner** — highest accuracy, highest F1, most stable across both dominant and minority classes.

### Per-Model Breakdown

**mDeBERTa-v3-base**
- Best performance across every metric
- Disentangled attention separates content and positional representations — crucial for code-switched text where word order doesn't follow a single language's grammar
- Most stable on ambiguous and hybrid tokens
- Reaches a lower evaluation loss faster than other models

**XLM-RoBERTa-base**
- Second-best overall
- Particularly strong on English (`lang2`) tokens and named entity tags
- SentencePiece tokenizer handles Romanized Hindi better than WordPiece
- Benefits from CC100 corpus scale (2.5TB across 100+ languages)

**mBERT**
- Solid mid-range baseline
- Competitive on high-frequency classes (`lang1`, `lang2`, `other`)
- Weaker on Romanized Hindi due to WordPiece tokenization and smaller pretraining corpus

**DistilBERT-multilingual-cased**
- Fastest inference — ~40% fewer parameters than mBERT
- Lowest accuracy and F1 across the board
- Struggles with code-switching complexity, morphological mixing, and low-frequency tags
- Trade-off: use it if deployment speed matters more than accuracy (e.g., mobile or real-time chat)

### Per-Class Observations

- High-support classes (`lang1`, `lang2`, `other`): F1 scores range from **0.75 to 0.98** across models — all models handle these well
- Low-support classes (`mixed`, `fw`, `unk`, `ixed`, `nk`): **F1 ≈ 0.00 across all models** — not enough training examples to learn reliable patterns
- `lang2` recall is consistent but tight (0.78–0.79 range), reflecting genuine ambiguity in tokens that appear in both languages

---

## Linguistic Challenges

These are the specific phenomena that made this task hard, regardless of model choice:

- **Rare label classes** — Tags like `ixed` and `nk` appear fewer than 7 times in the entire dataset. No model learned anything useful for these.

- **Cross-language ambiguous tokens** — Words like *is*, *to*, *me*, *or* are valid in both Hindi and English. Classification depends entirely on context, not spelling.

- **Inconsistent Roman transliteration of Hindi** — The same Hindi word appears as *nahi*, *nai*, *naahi*, *kyaaa*, *kyaah* etc. This breaks tokenization consistency, especially for smaller models.

- **Morphologically hybrid words** — *planning karna*, *confirm hua* mix English stems with Hindi morphology. These don't appear in standard pretraining corpora, so models produce inconsistent predictions for them. The English (`e`) class F1 scores dipped noticeably across models because of this: **0.82 → 0.80 → 0.75 → 0.77**.

---

## Key Takeaways

- **Model depth and pretraining scale matter** — mDeBERTa-v3 and XLM-RoBERTa consistently outperform shallower models on code-switched text
- **Disentangled attention is genuinely useful here** — Separating content and position representations helps when word order is inconsistent across languages
- **Class imbalance is the main bottleneck for rare tags** — Architecture alone can't fix zero-shot performance on labels with near-zero support
- **DistilBERT is a reasonable choice only if speed is the constraint** — It's meaningfully worse on every accuracy metric
- **Romanized Hindi normalization is an open problem** — Orthographic inconsistency hurts all models and needs dedicated preprocessing or data augmentation

---

## Authors

- **Mahesh Babu Vishnumolakala** — Lovely Professional University, Phagwara, India
- **Pavan Venkat Kumar Doddavarapu** — Lovely Professional University, Phagwara, India
- **Sasivardhan Mangira** — Lovely Professional University, Phagwara, India
- **Enjula Uchoi** — Lovely Professional University, Phagwara, India

---

*Built with HuggingFace Transformers, PyTorch, and the `AutoModelForTokenClassification` pipeline.*
