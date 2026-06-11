import pandas as pd
from sklearn.metrics import accuracy_score, log_loss as sklearn_log_loss


def calculate_accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def calculate_log_loss(y_true, y_proba, labels) -> float:
    return float(sklearn_log_loss(y_true, y_proba, labels=labels))


def calculate_multiclass_brier_score(y_true, y_proba, labels) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    total = 0
    for actual, probs in zip(y_true, y_proba):
        actual_vector = [0] * len(labels)
        actual_vector[label_to_index[actual]] = 1
        total += sum((prob - actual_value) ** 2 for prob, actual_value in zip(probs, actual_vector))
    return float(total / len(y_true))


def calculate_draw_calibration(y_true, draw_probabilities) -> pd.DataFrame:
    buckets = [
        (0.00, 0.20, "0.00-0.20"),
        (0.20, 0.25, "0.20-0.25"),
        (0.25, 0.30, "0.25-0.30"),
        (0.30, 0.35, "0.30-0.35"),
        (0.35, 0.40, "0.35-0.40"),
        (0.40, 1.01, "0.40+"),
    ]
    rows = []
    df = pd.DataFrame({"actual": list(y_true), "draw_probability": list(draw_probabilities)})
    for low, high, label in buckets:
        bucket_df = df[(df["draw_probability"] >= low) & (df["draw_probability"] < high)]
        rows.append(
            {
                "bucket": label,
                "count": len(bucket_df),
                "average_predicted_draw_probability": bucket_df["draw_probability"].mean() if not bucket_df.empty else 0.0,
                "actual_draw_rate": (bucket_df["actual"] == "D").mean() if not bucket_df.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)

