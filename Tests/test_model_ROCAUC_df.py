import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create synthetic dataset
customer_df = pd.DataFrame({
    "CustomerID": [1, 2, 3, 4, 5, 6],
    "FirstPurchase": [1, 1, 1, 1, 1, 1],
    "LastPurchase": [10, 20, 30, 40, 50, 60],
    "CustomerHealth": ["Active", "Active", "Churned", "Active", "Churn Risk", "Churned"],
    "CustomerHealth_num": [0, 0, 2, 0, 1, 2],
    "DaysSinceLastPurchase": [5, 10, 20, 30, 40, 50],
    "TotalNoOfOrders": [1, 2, 3, 4, 5, 6],
    "TotalSpend": [10, 20, 30, 40, 50, 60],
    "AvgSpend": [10, 10, 10, 10, 10, 10],
    "MaxSpend": [10, 20, 30, 40, 50, 60],
    "NoOfUniqueProducts": [1, 2, 3, 4, 5, 6],
    "ChurnBinary": [0, 0, 1, 0, 0, 1]
})

# Import models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=1)
}

def model_ROCAUC_df(customer_df, models):
    # Features and target
    X = customer_df.drop(['CustomerID', 'FirstPurchase', 'LastPurchase', 'CustomerHealth', 'CustomerHealth_num', 'ChurnBinary'], axis=1)
    y = customer_df['ChurnBinary']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    results = []

    # Loop through models
    for name, model in models.items():
        scaling_models = ["Logistic Regression", "SVM"]

        # Modelling pipeline with scaling for models that require it and evaluate performance using ROC AUC 
        if name in scaling_models:

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            try:
                pipeline.fit(X_train, y_train)
            except Exception as e:
                print(f"Error fitting {name}: {e}")

            y_test_probs = pipeline.predict_proba(X_test)[:, 1]
            model_fpr, model_tpr, _ = roc_curve(y_test, y_test_probs)
            model_auc = auc(model_fpr, model_tpr)

            results.append([name,model_fpr, model_tpr, model_auc])

       # Evaluate performance using ROC AUC without scaling for models that do not require it
        else:
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"Error fitting {name}: {e}")

            y_test_probs = model.predict_proba(X_test)[:, 1]
            model_fpr, model_tpr, _ = roc_curve(y_test, y_test_probs)
            model_auc = auc(model_fpr, model_tpr)

            results.append([name,model_fpr, model_tpr, model_auc])


    # Create results DataFrame
    results_df = pd.DataFrame(
         results,
         columns=["Model", "model_fpr", "model_tpr", "model_auc"]
     )

    return results_df



def test_model_ROCAUC_df():

    results = model_ROCAUC_df(customer_df, models)

    # Check output type
    assert isinstance(results, pd.DataFrame), "Output is not a DataFrame"

    # Check number of rows = number of models
    assert len(results) == len(models), "Number of models does not match output rows"

    # Check required columns
    expected_cols = [
        "Model",
        "model_fpr",
        "model_tpr",
        "model_auc"
    ]

    for col in expected_cols:
        assert col in results.columns, f"Missing column: {col}"

    # Check values are numeric
    for val in results["model_fpr"]:
        assert isinstance(val, np.ndarray), "model_fpr values are not arrays"

    for val in results["model_tpr"]:
        assert isinstance(val, np.ndarray), "model_tpr values are not arrays"
        
    for val in results["model_auc"]:
        assert isinstance(val, float), "model_auc values are not floats"

    print("model_ROCAUC_df tests passed")


if __name__ == "__main__":
    test_model_ROCAUC_df()
