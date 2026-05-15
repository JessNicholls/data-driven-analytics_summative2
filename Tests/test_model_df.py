import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
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
    "NoOfUniqueProducts": [1, 2, 3, 4, 5, 6]
})

# Import models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=1)
}

# Create modelling function
def model_df(customer_df, models):
    # Features and target
    X = customer_df.drop(['CustomerID', 'FirstPurchase', 'LastPurchase', 'CustomerHealth', 'CustomerHealth_num'], axis=1)
    y = customer_df['CustomerHealth_num']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Create empty list to store results
    results = []

    # Loop through models
    for name, model in models.items():
        scaling_models = ["Logistic Regression", "KNN", "SVM"]
        
        # Create modelling pipeline with scaling for models that require it and evaluate performance
        if name in scaling_models:

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            try:
                pipeline.fit(X_train, y_train)
            except Exception as e:
                print(f"Error fitting {name}: {e}")
            #pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            cv_scores = cross_val_score(pipeline, X, y, cv=2, scoring='accuracy')

        else:
            # Fit model and evaluate performance without scaling
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"Error fitting {name}: {e}")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            cv_scores = cross_val_score(model, X, y, cv=2, scoring='accuracy')

        # Generate Summary Scores
        train_score = accuracy_score(y_train, y_train_pred) * 100
        test_score = accuracy_score(y_test, y_test_pred) * 100
        cv_mean = cv_scores.mean() * 100

        # Store results in df
        results.append([
            name,
            round(train_score, 2),
            round(test_score, 2),
            round(cv_mean, 2),
            round(train_score - test_score, 2)

        ])

    # Create results DataFrame from list
    results_df = pd.DataFrame(
        results,
        columns=["Model", "Train Accuracy (%)", "Test Accuracy (%)", "CV Accuracy (%)", "Overfit (%)"]
    )

    # Return the results DataFrame
    return results_df


def test_model_df():

    results = model_df(customer_df, models)

    # Check output type
    assert isinstance(results, pd.DataFrame), "Output is not a DataFrame"

    # Check number of rows = number of models
    assert len(results) == len(models), "Number of models does not match output rows"

    # Check required columns
    expected_cols = [
        "Model",
        "Train Accuracy (%)",
        "Test Accuracy (%)",
        "CV Accuracy (%)",
        "Overfit (%)"
    ]

    for col in expected_cols:
        assert col in results.columns, f"Missing column: {col}"

    # Check values are numeric
    assert results["Train Accuracy (%)"].dtype != "object"
    assert results["Test Accuracy (%)"].dtype != "object"

    print("model_df tests passed")


if __name__ == "__main__":
    test_model_df()
