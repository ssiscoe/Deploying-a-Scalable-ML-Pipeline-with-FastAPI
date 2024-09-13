# Model Card

For additional information, see the Model Card paper: [https://arxiv.org/pdf/1810.03993.pdf](https://arxiv.org/pdf/1810.03993.pdf)

## Model Details

**Model Name:** RandomForestClassifier

**Model Description:** 
The RandomForestClassifier is an ensemble machine learning model that combines multiple decision trees to improve classification performance. Each tree is trained on a subset of the data and features, and the final prediction is made by aggregating the predictions of all trees.

**Model Type:** Classification

**Model Architecture:** 
The model consists of 100 decision trees, trained using the RandomForestClassifier algorithm from scikit-learn. It predicts whether an individual earns more than $50,000 per year based on various demographic features.

## Intended Use

The model is designed to predict whether an individual’s annual income exceeds $50,000 based on their demographic information. It can be used by organizations to understand income distribution or to tailor services based on income levels. It should be used with caution to ensure fair treatment and avoid bias.

## Training Data

**Data Source:** 
The model was trained using the `census.csv` dataset, which contains census demographic information.

**Data Description:** 
- **Number of Instances:** 32,561 (training set)
- **Number of Features:** 8 categorical features
- **Label:** `salary` (Binary classification: >50K or <=50K)
- **Features:** 
  - `workclass`
  - `education`
  - `marital-status`
  - `occupation`
  - `relationship`
  - `race`
  - `sex`
  - `native-country`

**Data Preprocessing:** 
The features were encoded using OneHotEncoder, and the labels were binarized using LabelBinarizer. The data was split into training and test sets.

## Evaluation Data

**Evaluation Source:** 
The performance of the model was evaluated on a held-out test set from the `census.csv` dataset.

**Evaluation Description:** 
- **Number of Instances:** 8,152 (test set)

## Metrics

**Metrics Used:**
- **Precision:** Measures the accuracy of positive predictions.
- **Recall:** Measures the model’s ability to identify all relevant instances.
- **F1 Score:** The harmonic mean of precision and recall.

**Performance on Test Data:**
- **Precision:** 0.7376
- **Recall:** 0.6288
- **F1 Score:** 0.6789

**Performance on Categorical Slices:**

| Workclass        | Precision | Recall | F1 Score | Count |
|------------------|-----------|--------|----------|-------|
| Federal-gov      | 0.5000    | 0.4762 | 0.4878   | 188   |
| Local-gov        | 0.8197    | 0.7353 | 0.7752   | 430   |
| Never-worked     | 1.0000    | 1.0000 | 1.0000   | 1     |
| Private          | 0.7500    | 0.8182 | 0.7826   | 4,595 |
| Self-emp-inc     | 0.7381    | 0.6245 | 0.6766   | 201   |
| Self-emp-not-inc | 0.7500    | 0.8182 | 0.7826   | 495   |
| State-gov        | 0.7333    | 0.5203 | 0.6087   | 248   |

## Ethical Considerations

The model should be used with caution to avoid perpetuating or amplifying biases present in the data. It is important to ensure that predictions do not unfairly discriminate against individuals based on their demographic attributes. Users of this model should be aware of the potential for bias and take steps to mitigate it.

## Caveats and Recommendations

- **Caveats:**
  - The model's performance may vary across different demographic groups, and it may not generalize well to individuals outside the training data's demographic distribution.
  - The model may not perform optimally on data that significantly deviates from the distribution of the training data.

- **Recommendations:**
  - Regularly update the model with new data to maintain accuracy and relevance.
  - Implement fairness and bias mitigation strategies to ensure equitable outcomes.
  - Consider complementing the model with additional features or data sources for improved predictions.
