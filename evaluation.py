import pandas as pd
from sklearn.metrics import accuracy_score, log_loss as sklearn_log_loss


def calculate_accuracy(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(accuracy_score(y_true, y_pred))


def calculate_log_loss(y_true, y_proba, labels=None) -> float:
    labels = labels or ["H", "D", "A"]
    if len(y_true) == 0:
        return 0.0
    return float(sklearn_log_loss(y_true, y_proba, labels=labels))


def calculate_multiclass_brier_score(y_true, y_proba, labels=None) -> float:
    labels = labels or ["H", "D", "A"]
    if len(y_true) == 0:
        return 0.0
    label_to_index = {label: index for index, label in enumerate(labels)}
    total = 0
    for actual, probs in zip(y_true, y_proba):
        actual_vector = [0] * len(labels)
        actual_vector[label_to_index[actual]] = 1
        total += sum((prob - actual_value) ** 2 for prob, actual_value in zip(probs, actual_vector))
    return float(total / len(y_true))


def calculate_expected_calibration_error(y_true, y_proba, labels=None, n_bins: int = 10) -> float:
    labels = labels or ["H", "D", "A"]
    if len(y_true) == 0:
        return 0.0
    df = pd.DataFrame(y_proba, columns=labels)
    df["actual"] = list(y_true)
    df["confidence"] = df[labels].max(axis=1)
    df["predicted"] = df[labels].idxmax(axis=1)
    df["correct"] = df["predicted"] == df["actual"]
    ece = 0.0
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        mask = (df["confidence"] >= lower) & (df["confidence"] <= upper if index == n_bins - 1 else df["confidence"] < upper)
        bucket = df[mask]
        if bucket.empty:
            continue
        weight = len(bucket) / len(df)
        ece += weight * abs(float(bucket["confidence"].mean()) - float(bucket["correct"].mean()))
    return float(ece)


def calculate_class_rates(y_true, y_proba, labels=None) -> dict:
    labels = labels or ["H", "D", "A"]
    if len(y_true) == 0:
        return {
            "actual_home_rate": 0.0,
            "actual_draw_rate": 0.0,
            "actual_away_rate": 0.0,
            "avg_pred_home_prob": 0.0,
            "avg_pred_draw_prob": 0.0,
            "avg_pred_away_prob": 0.0,
            "draw_calibration_gap": 0.0,
            "home_calibration_gap": 0.0,
            "away_calibration_gap": 0.0,
        }
    df = pd.DataFrame(y_proba, columns=labels)
    actual = pd.Series(list(y_true))
    values = {}
    for label, name in [("H", "home"), ("D", "draw"), ("A", "away")]:
        actual_rate = float((actual == label).mean())
        pred_rate = float(df[label].mean()) if label in df.columns else 0.0
        values[f"actual_{name}_rate"] = actual_rate
        values[f"avg_pred_{name}_prob"] = pred_rate
        values[f"{name}_calibration_gap"] = pred_rate - actual_rate
    return values


def calculate_prediction_metrics(y_true, y_proba, labels=None) -> dict:
    labels = labels or ["H", "D", "A"]
    if len(y_true) == 0:
        metrics = calculate_class_rates(y_true, y_proba, labels)
        metrics.update({"accuracy": 0.0, "log_loss": 0.0, "brier_score": 0.0, "ece": 0.0})
        return metrics
    proba_df = pd.DataFrame(y_proba, columns=labels)
    y_pred = proba_df.idxmax(axis=1).tolist()
    metrics = {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "log_loss": calculate_log_loss(y_true, proba_df[labels].to_numpy(), labels),
        "brier_score": calculate_multiclass_brier_score(y_true, proba_df[labels].to_numpy(), labels),
        "ece": calculate_expected_calibration_error(y_true, proba_df[labels].to_numpy(), labels),
    }
    metrics.update(calculate_class_rates(y_true, proba_df[labels].to_numpy(), labels))
    return metrics


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
