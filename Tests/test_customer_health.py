import pandas as pd
customer_df = pd.DataFrame({
    "DaysSinceLastPurchase": [10, 20, 30, 40, 50, 60, 70, 80]
})

def customer_health(recency):
    if recency <= customer_df["DaysSinceLastPurchase"].quantile(0.65):
        return "Active"
    elif recency <= customer_df["DaysSinceLastPurchase"].quantile(0.71):
        return "Churn Risk"
    else:
        return "Churned"

def test_customer_health():
    q65 = customer_df["DaysSinceLastPurchase"].quantile(0.65)
    q71 = customer_df["DaysSinceLastPurchase"].quantile(0.71)

    # Active
    assert customer_health(q65 - 1) == "Active"
    assert customer_health(q65) == "Active"

    # Churn Risk
    assert customer_health((q65 + q71) / 2) == "Churn Risk"
    assert customer_health(q71) == "Churn Risk"

    # Churned
    assert customer_health(q71 + 1) == "Churned"

    print("All tests passed")

# Run test
if __name__ == "__main__":
    test_customer_health()