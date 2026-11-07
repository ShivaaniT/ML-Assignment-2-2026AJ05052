# Bank Marketing Subscription Prediction Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on customer demographics and previous marketing campaign information.

Five classification algorithms are implemented and evaluated using multiple performance metrics. A Streamlit web application is developed to demonstrate and compare the trained models.

## 2. Dataset Description

**Dataset Name:** Bank Marketing Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing

- Total instances: 45,211
- Input features: 16 originally
- Features used: 15 (duration excluded to avoid data leakage)
- Target variable: y
- Problem type: Binary Classification
- Target classes: Yes and No

The dataset contains customer demographic information, financial details and marketing campaign information.

## 3. GitHub Repository Link

https://github.com/ShivaaniT/ML-Assignment-2-2026AJ05052

## 4. Models Used

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

### Evaluation Metrics

The following metrics were calculated:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8268 | 0.7465 | 0.3490 | 0.5548 | 0.4285 | 0.3450 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

### Observations on Model Performance

## Logistic Regression

Logistic Regression achieved an accuracy of 75.48% and an AUC score of 0.7722. It obtained a recall of 0.6238, indicating that it identified a relatively high proportion of actual positive cases. However, its precision of 0.2662 indicates a considerable number of false-positive predictions.

## Decision Tree

Decision Tree achieved an accuracy of 82.68% and an F1 score of 0.4285, the highest among all five models. It provided a reasonable balance between precision (0.3490) and recall (0.5548). Its MCC score of 0.3450 was also the highest, indicating the strongest overall correlation between predicted and actual classifications among the evaluated models.

## K-Nearest Neighbors (KNN)

KNN achieved the highest accuracy of 88.74% and precision of 0.5529. However, its recall was only 0.1975, meaning that it missed a substantial proportion of actual positive cases. Its F1 score of 0.2911 was the lowest among the five models, indicating that its high accuracy did not translate into balanced classification performance.

## Naive Bayes

Naive Bayes achieved an accuracy of 84.52% and an F1 score of 0.4123. It obtained a recall of 0.4641 and precision of 0.3708, demonstrating moderate performance in identifying positive and negative cases. Its AUC score of 0.7514 indicates reasonable class discrimination.

## Support Vector Machine

Support Vector Machine achieved an accuracy of 75.78% and an AUC score of 0.7718, which was the second-highest AUC among the evaluated models. Its recall of 0.6200 was close to Logistic Regression, indicating its ability to identify positive cases. However, its precision of 0.2684 suggests a relatively high number of false positives.

### Overall Winner

The best-performing model will be selected based on the actual evaluation results, considering F1 Score, AUC, Recall and MCC.

**Best Model:** Decision Tree

**Reason:** It achieved the highest F1 score of 0.4285 and MCC of 0.3450, demonstrating a better balance between precision and recall compared to the other models.

## 5. Streamlit Application

The application provides:

- CSV test dataset upload
- Classification model selection
- Six evaluation metrics
- Confusion matrix
- Classification report

The application loads previously trained models and evaluates the uploaded test dataset without retraining.

**Live Application Link:** https://ml-assignment-2-2026aj05052.streamlit.app/

## 6. Conclusion

Among the five classification algorithms, Decision Tree emerged as the best overall model based on the F1 score and Matthews Correlation Coefficient (MCC). It achieved the highest F1 score of 0.4285 and MCC of 0.3450, demonstrating a better balance between precision and recall compared to the other models.

Although KNN achieved the highest accuracy of 88.74%, its low recall of 0.1975 indicates that it failed to identify many actual positive cases. Since the objective is to predict customer subscription, identifying positive cases is important. Therefore, Decision Tree is selected as the overall winner for this dataset.
