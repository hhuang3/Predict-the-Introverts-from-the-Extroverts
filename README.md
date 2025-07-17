# Predict-the-Introverts-from-the-Extroverts


The dataset contains missing values in both numerical and categorical features `(e.g., Time_spent_Alone, Stage_fear, etc.).` These were handled appropriately prior to or during model training to ensure model integrity.

Categorical features `(Stage_fear, Drained_after_socializing, Personality)` were label-encoded into numeric format for compatibility with the XGBoost model.

As revealed during EDA, the target variable (Personality) is imbalanced, with the majority class (Extrovert) appearing nearly three times more often than the minority class (Introvert). To address this problem, the `scale_pos_weight parameter` was introduced:

`cale_pos_weight = 2740 / 965 ≈ 2.84`

```python
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2.5, 3]
}
```

A baseline XGBoost model was trained to serve as a reference. As shown in the results, the baseline model achieved strong performance with high accuracy and well-balanced precision and recall across both classes. This is largely due to XGBoost’s built-in regularization and robust tree-splitting mechanisms.

While hyperparameter tuning provided slight improvements, the baseline model’s solid performance suggests that the dataset is inherently learnable and well-structured, allowing XGBoost to generalize effectively even with minimal tuning.







