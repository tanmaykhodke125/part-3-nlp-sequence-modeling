# Part 3: NLP and Sequence Modeling Mini Project

## Overview
This project builds a **Natural Language Processing (NLP) pipeline** to classify customer support messages as **positive, neutral, or negative** sentiment. It compares traditional text vectorization (TF-IDF) with deep learning approaches (LSTM).

## Dataset
- **File:** `customer_support_text_classification.csv`
- **Source:** [Google Drive Folder](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)
- **Target Column:** `sentiment_label` (positive / neutral / negative)
- **Input Column:** `customer_message` (raw text)

## Steps Performed
1. **Dataset Understanding** — Loaded data, checked class distribution, average message length
2. **Text Preprocessing** — Lowercasing, removed special characters, stopword removal, tokenization
3. **Text Vectorization** — TF-IDF with 5000 features
4. **Baseline Model** — Logistic Regression with TF-IDF
5. **Sequence Model** — LSTM with Embedding layer
6. **Reflection** — RNNs, LSTMs, Attention, Transformers

## Why Text Must Be Converted to Vectors
Neural networks and ML models only understand numbers, not words. Vectorization converts text into numerical arrays that the model can process. Without this step, we cannot feed text into any model.

## Results
See `results/model_evaluation.csv` for accuracy and F1-score comparison.
See `results/sample_predictions.txt` for sample message predictions.

## LSTM Architecture
```
Input (sequence of token IDs)
  ↓
Embedding Layer (5000 vocab, 64 dimensions)
  ↓
LSTM(64 units)
  ↓
Dense(3, Softmax)  ← 3 sentiment classes
```
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam

## Reflection on Sequence Models

### Why Do RNNs Struggle with Long-Term Dependencies?
RNNs process text one word at a time and pass a "hidden state" forward. For long sentences, information from early words gets diluted or forgotten by the time the model reaches later words — this is called the **vanishing gradient problem**.

### How Do LSTMs Help with Memory?
LSTMs (Long Short-Term Memory) add special "gates" — input gate, forget gate, output gate — that control what information to keep, discard, or pass forward. This allows LSTMs to selectively remember important context even across long sequences.

### What Does Attention Solve?
Attention mechanisms allow a model to look back at ALL previous words at every step, not just the most recent hidden state. This means:
- Important words from earlier in the sentence can directly influence the current prediction
- The model learns which words are most relevant to attend to
- This solved many problems in machine translation and text summarization

### Why Are Transformers Important in Modern NLP and Generative AI?
Transformers use **self-attention** and process ALL words in parallel (not sequentially like RNNs). This makes them:
- **Faster to train** (parallelizable)
- **Better at long-range dependencies**
- **Scalable** to massive datasets
- The foundation of GPT, BERT, ChatGPT, and all modern large language models (LLMs)

## How to Run
1. Open `notebook.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `customer_support_text_classification.csv` from the dataset folder
3. Run all cells from top to bottom

## Requirements
See `requirements.txt`
