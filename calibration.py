import pandas as pd


def _probability_frame(y_true, y_proba, labels):
    df = pd.DataFrame(y_proba, columns=labels)
    df["actual"] = list(y_true)
    return df


def create_confidence_calibration_bins(y_true, y_proba, labels=None, n_bins: int = 10) -> pd.DataFrame:
    labels = labels or ["H", "D", "A"]
    df = _probability_frame(y_true, y_proba, labels)
    if df.empty:
        return pd.DataFrame(
            columns=["bin_lower", "bin_upper", "count", "avg_confidence", "accuracy", "calibration_gap"]
        )
    probabilities = df[labels]
    df["confidence"] = probabilities.max(axis=1)
    df["predicted"] = probabilities.idxmax(axis=1)
    df["correct"] = df["predicted"] == df["actual"]
    edges = [i / n_bins for i in range(n_bins + 1)]
    rows = []
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        mask = (df["confidence"] >= lower) & (df["confidence"] <= upper if index == n_bins - 1 else df["confidence"] < upper)
        bucket = df[mask]
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": int(len(bucket)),
                "avg_confidence": float(bucket["confidence"].mean()) if not bucket.empty else 0.0,
                "accuracy": float(bucket["correct"].mean()) if not bucket.empty else 0.0,
                "calibration_gap": float(bucket["confidence"].mean() - bucket["correct"].mean()) if not bucket.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def create_class_probability_bins(y_true, class_probabilities, positive_class: str, bins) -> pd.DataFrame:
    df = pd.DataFrame({"actual": list(y_true), "probability": list(class_probabilities)})
    rows = []
    for index, lower in enumerate(bins[:-1]):
        upper = bins[index + 1]
        label = f"{lower:.2f}-{upper:.2f}" if upper < 1 else f"{lower:.2f}+"
        mask = (df["probability"] >= lower) & (df["probability"] <= upper if index == len(bins) - 2 else df["probability"] < upper)
        bucket = df[mask]
        rows.append(
            {
                "bin_label": label,
                "count": int(len(bucket)),
                "avg_predicted_probability": float(bucket["probability"].mean()) if not bucket.empty else 0.0,
                "actual_rate": float((bucket["actual"] == positive_class).mean()) if not bucket.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def create_draw_calibration_table(y_true, draw_probabilities) -> pd.DataFrame:
    result = create_class_probability_bins(
        y_true,
        draw_probabilities,
        positive_class="D",
        bins=[0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0],
    )
    result = result.rename(
        columns={
            "avg_predicted_probability": "avg_predicted_draw_probability",
            "actual_rate": "actual_draw_rate",
        }
    )
    result["calibration_gap"] = result["avg_predicted_draw_probability"] - result["actual_draw_rate"]
    return result
