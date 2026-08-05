# DL-and-Gen-AI-Project


# Multi-Label Emotion Classification using Deep Learning

## Project Overview
This project implements a robust multi-label text classification system designed to detect fine-grained emotions in English text. Unlike standard sentiment analysis which classifies text as merely positive or negative, this system classifies entries into five overlapping emotion categories: **Anger, Fear, Joy, Sadness, and Surprise**.

The core challenge is **Multi-Label Classification**, meaning a single text entry can convey multiple emotions simultaneously (e.g., *Fear* + *Sadness*). To solve this, we implemented and benchmarked three distinct modeling paradigms:
1.  **Classical Machine Learning:** Baseline models using TF-IDF vectors with SVM, Logistic Regression, and Naive Bayes.
2.  **Custom Deep Learning:** Recurrent Neural Networks (Bi-LSTM, Stacked GRU) and Convolutional Neural Networks (TextCNN) trained from scratch using PyTorch.
3.  **Transfer Learning:** State-of-the-art Transformer models (RoBERTa, BERT, DistilBERT) fine-tuned to achieve maximum accuracy and generalization.

## Key Features
* **Multi-Label Classification:** Capable of predicting multiple co-occurring emotions (e.g., *Fear* + *Sadness*) for a single text entry.
* **Hybrid Architecture:** Implements both sequential (Bi-LSTM, Stacked GRU) and spatial (TextCNN) deep learning models built from scratch in PyTorch.
* **Attention Mechanisms:** Custom Bidirectional LSTM augmented with Self-Attention to dynamically weight important emotional keywords.
* **Performance:** Fine-tuned **RoBERTa** model achieving a Kaggle Leaderboard score of **0.764**, outperforming classical baselines by over 30%.
* **Experiment Tracking:** Fully integrated with **Weights & Biases (W&B)** for real-time monitoring of training loss, F1 scores, and hyperparameter sweeping.

## Model Performance
We evaluated 10+ architectures using the **Macro F1-Score** to account for class imbalance.

| Model Category | Architecture | Val Macro F1 | Kaggle Score |
| :--- | :--- | :--- | :--- |
| **Transformer** | **RoBERTa (Best Model)** | **0.771** | **0.764** |
| Transformer | BERT | 0.747 | 0.735 |
| Transformer | DistilBERT | 0.715 | 0.705 |
| **Custom DL** | **TextCNN** | **0.722** | **0.700** |
| Custom DL | Bi-LSTM + Attention | 0.709 | 0.707 |
| Custom DL | Stacked GRU | 0.701 | 0.685 |
| **Classical** | **SVM (Support Vector Machine)** | **0.687** | **0.702** |
| Classical | Logistic Regression | 0.436 | 0.444 |

## Setup & Installation

### Prerequisites
* Python 3.11+
* CUDA-enabled GPU (Recommended for training LSTMs and Transformers)

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2.Install Dependencies
```bash

pip install -r requirements.txt
```

## Usage Guide

### Phase 1: Data Analysis & Preprocessing
Run `notebooks/EDA.ipynb` to visualize the dataset.
* **Input:** `data/train.csv`
* **Output:** Word clouds, Class distribution plots, Text statistics.

### Phase 2: Training Custom Models (RNN/CNN)
To train the Bi-LSTM or TextCNN models:
1.  Open `scripts/Training/Bi-LSTM.ipynb`.
2.  Ensure `vocab.json` is generated during the first run.
3.  Execute all cells to train the model.
4.  **Inference:** The notebook automatically saves the best weights as `bilstm_model.pth` and the vocabulary as `vocab.json` to the `models/` directory.

### Phase 3: Fine-Tuning Transformers
To reproduce the best results (RoBERTa):
1.  Open `scripts/Training/RoBERTa.ipynb`.
2.  Set `MODEL_CHECKPOINT = 'roberta-base'`.
3.  Run the training loop (utilizing the Hugging Face `Trainer` API).

## Technologies Used

* **Deep Learning Frameworks:** PyTorch, Hugging Face Transformers
* **Machine Learning Libraries:** Scikit-learn (SVM, Naive Bayes, Logistic Regression)
* **Natural Language Processing:** NLTK (Tokenization, Lemmatization, Stop words), Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn, WordCloud
* **Experiment Tracking:** Weights & Biases (W&B)

## Deployment

The project includes an interactive web application built with **Streamlit** to demonstrate the RoBERTa model in real-time.

### Live Demo
**[Click here to access the Live App](https://dl-and-gen-ai-project-deployment.streamlit.app/)** 

### ☁️ Deployment Architecture
Due to the large size of Transformer models (>450MB), standard GitHub pushing is disabled. This project uses **Git LFS (Large File Storage)** and a dedicated orphan branch (`deploy-app`) to host the production assets.

#### How to Run the App Locally
If you clone the repository, standard `git clone` will only download pointers for the model files. To get the actual weights:

1.  **Install Git LFS:**
    ```bash
    git lfs install
    ```
2.  **Fetch the Deployment Branch:**
    ```bash
    git fetch origin deploy-app
    git checkout deploy-app
    ```
3.  **Pull the Large Files:**
    ```bash
    git lfs pull
    ```
4.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```

#### Hosting on Streamlit Cloud
The app is deployed directly from the `deploy-app` branch.
* **Configuration:** `safetensors` model format is used for faster, safer loading.
* **Caching:** `@st.cache_resource` is implemented to load the BERT model only once, preventing memory overflows on the free tier.
