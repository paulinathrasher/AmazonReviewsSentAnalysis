# Amazon Reviews Sentiment Analysis

## Overview
This project uses transformer-based sentence embeddings to classify Amazon product reviews as positive or negative. The analysis demonstrates how Natural Language Processing (NLP) techniques can help businesses monitor customer sentiment at scale, enabling data-driven product improvements and better customer experience. Multiple models were implemented, starting with a simple centroid-based classifier and progressing to a logistic regression classifier with cross-validation, achieving 90% accuracy.

## Dataset
- Source: Mock Amazon product reviews
- Size: 241 reviews (expanded from 161 in initial iteration)
- Features:
  - Review Text (processed into embeddings)
  - Rating (1–5 scale, converted to binary labels: ≤3 = negative, >3 = positive)
  - Product ID (not used for modeling)
- Class Distribution:
  - Positive: Majority
  - Negative: Minority (class imbalance addressed through stratified cross-validation)

## Methods
### Data Preprocessing
- Embeddings: Generated using Hugging Face’s "sentence-transformers/all-MiniLM-L6-v2"
- Feature Representation:
  - Mean pooling of token embeddings → fixed-length review vectors
  - L2 normalization tested to ensure fair cosine similarity comparisons
- Binary Labels: Based on review ratings (>3 positive, ≤3 negative)
- Train-Test Split: 80% training, 20% testing (later replaced with cross-validation)
- Cross-Validation: Stratified 5-fold cross-validation to reduce bias and variance

### Models Implemented
| Model | Accuracy | ROC-AUC | Notes |
|-------|----------|---------|-------|
| Centroid (Single Split) | 77.5% | 0.77 | Baseline; cosine similarity to positive/negative centroids |
| Centroid + L2 Normalization | 85.7% | 0.86 | Improved similarity comparisons |
| Centroid + 5-Fold Cross Validation | 78.4% | 0.79 | Reduced variance, standard deviation 0.066 |
| Logistic Regression + Cross Validation | **90.0%** | **0.89** | Best performance; learned weights from embeddings |

## Results
- Business Insights:
  - Logistic regression outperformed distance-based models, suggesting that learning patterns from embeddings captures sentiment more effectively.
  - Transformer-based embeddings improved semantic understanding compared to traditional TF-IDF or bag-of-words methods.
- Key Metrics (Best Model):
  - Accuracy: 90.04%
  - ROC-AUC: 0.89 → strong ability to distinguish positive vs. negative reviews
  - Error Analysis: 9 false positives, 15 false negatives across 5 folds

## Next Steps
- Expand dataset further to improve generalization
- Experiment with other classifiers (e.g., Support Vector Machines, Random Forests)
- Introduce a neutral/mixed sentiment class
- Investigate explainability with SHAP or LIME for better business insights

## How to Run
1. Clone the repository:
