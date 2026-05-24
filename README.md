# Strategic Customer Retention via Deep Learning

Multi-class classification of subscriber retention strategies using a deep learning ensemble. Built for a streaming service dataset of 10,000+ labeled customer records.

## Problem

Streaming platforms lose revenue when they apply the wrong retention strategy to the wrong customer. Blanket discounting is costly. Ignoring high-risk subscribers accelerates churn. This project frames retention as a supervised classification problem: given a customer's behavioral and demographic profile, predict which of four strategic actions to take.

**Classes:**
- 0: Monitor (low churn risk, no action needed)
- 1: Re-Engage (declining engagement, targeted outreach)
- 2: Discount (price-sensitive, offer incentive)
- 3: Terminate (low lifetime value, do not retain)

## Approach

### Feature Engineering
One-hot encoding for categorical variables. Six interaction features constructed from engagement score, tenure, income, age, and device count to capture non-linear relationships.

### Model Architecture
5-layer feed-forward neural network with batch normalization and dropout at each layer. Input layer scales to the number of engineered features. Output layer uses softmax over 4 classes.

### Hyperparameter Search
Multiple configurations tested across activation functions (ELU, ReLU), dropout rates, batch sizes, and learning rates. Top 3 configurations selected by validation accuracy for ensemble construction.

### Ensemble
Majority-vote ensemble across 3 independently trained models. Ensemble used if it matches or exceeds the best single model on the held-out validation set.

### Training Setup
- Optimizer: Adam (lr=0.001)
- Loss: Sparse categorical crossentropy
- Early stopping on validation accuracy (patience=25)
- Learning rate reduction on plateau (factor=0.3, patience=8)
- Train/validation split: 85/15, stratified

## Results

| Method | Validation Accuracy |
|---|---|
| Best single model | 90.41% |
| Majority-vote ensemble | 90%+ |

Per-class precision, recall, and F1 available in the classification report cell.

## Stack

Python, TensorFlow/Keras, scikit-learn, NumPy, pandas, SciPy

## Files

| File | Description |
|---|---|
| `customer-retention-deep-learning.ipynb` | Full pipeline: preprocessing, training, ensemble, predictions |
| `Project 01 - Data.csv` | Source data (not included, proprietary) |
| `Group 11.csv` | Output predictions for 10,000 unlabeled customers |

## Usage

```bash
# Run in Google Colab with GPU runtime (T4 recommended)
# Place data file in the same directory as the notebook
# Runtime > Run all
```

## Authors

Group 11, MSBA 585 Deep Learning, University of Washington Tacoma
