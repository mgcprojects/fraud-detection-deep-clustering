# Credit Card Fraud Detection via Deep Clustering & Neural Networks

A hybrid fraud detection framework that combines deep autoencoders, K-Means clustering, and cluster-specific neural networks to detect fraudulent transactions in highly imbalanced data.

## Overview

Credit card fraud detection is challenging because fraudulent transactions make up a tiny fraction of all activity (0.172% in this dataset). A single global classifier often struggles to capture the diversity of fraud patterns. This project takes a different approach: first learn a compressed representation of transaction behavior using an autoencoder, then segment transactions into behavioral clusters, and finally train specialized neural networks within each cluster.

Built for CS7357 (Neural Networks & Deep Learning) at Kennesaw State University.

## My Role

- Designed and implemented the **deep clustering pipeline**: autoencoder training, latent space extraction, K-Means segmentation, and cluster-specific MLP training
- Evaluated performance across multiple cluster configurations (k = 2, 4, 5, 10, 20, 50)
- Conducted comparative analysis against the SMOTE-only baseline

## Key Results

| Metric | Baseline (NN + SMOTE) | Deep Clustering (k=10) | Improvement |
|--------|----------------------|----------------------|-------------|
| F1-Score | 0.382 | 0.498 | +30% |
| PR-AUC | — | 0.756 | — |
| Precision | 0.250 | 0.356 | +42% |
| Recall | 0.808 | 0.827 | +2% |
| ROC-AUC | 0.951 | 0.920 | — |

The deep clustering approach at k=10 achieved the best balance between catching fraud and minimizing false alarms, significantly outperforming the SMOTE-only baseline on precision and F1 while maintaining comparable recall.

### Performance Across Cluster Configurations

| k | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
|---|---------|--------|-----------|--------|----------|
| 2 | 0.951 | 0.691 | 0.031 | 0.880 | 0.059 |
| 5 | 0.903 | 0.449 | 0.019 | 0.840 | 0.038 |
| **10** | **0.920** | **0.756** | **0.356** | **0.827** | **0.498** |
| 20 | 0.946 | 0.714 | 0.136 | 0.840 | 0.234 |
| 50 | 0.946 | 0.680 | 0.116 | 0.720 | 0.200 |

## Approach

1. **Data**: Kaggle Credit Card Fraud Detection dataset — 284,807 transactions, 492 fraudulent (0.172%)
2. **Preprocessing**: Z-score normalization on Time/Amount, chronological 70/15/15 split to prevent data leakage
3. **Baseline**: MLP (64→32→16) trained on SMOTE-balanced data with early stopping on validation AUC
4. **Deep Clustering**: Autoencoder learns latent transaction representations → K-Means clusters transactions by behavioral similarity → separate MLPs trained per cluster with local class weighting
5. **Inference**: New transaction assigned to nearest cluster centroid → corresponding cluster-specific model generates prediction
6. **Evaluation**: Precision, Recall, F1, ROC-AUC, PR-AUC, and MCC across all configurations

## Tech Stack

Python, TensorFlow, Scikit-learn, K-Means, Autoencoders, SMOTE (imbalanced-learn), Pandas, NumPy, Matplotlib, Seaborn

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/fraud_detection_deep_clustering.ipynb
```

The notebook runs end-to-end: data loading, preprocessing, baseline training, autoencoder training, clustering, cluster-specific model training, and evaluation.

## Team

- **Mauricio Gonzalez** – Deep clustering pipeline, autoencoder design, hybrid model integration
- **Sukumar Muthusamy** – Baseline NN + SMOTE implementation, evaluation metrics
- **Dhruv Shrivastava** – Hyperparameter tuning, optimization, performance analysis

## License

This project was developed collaboratively for CS7357 at Kennesaw State University.
