"""Reconstruct the original UCI Iris 150-row CSV from sklearn's bundled version.

scikit-learn v0.20+ fixes two erroneous entries in original UCI iris.data.
The original UCI measurements (0-indexed rows 34 & 37) are restored here.
See https://archive.ics.uci.edu/dataset/53/iris and sklearn load_iris docs.
"""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "species"]


def main() -> None:
    ds = load_iris()
    df = pd.DataFrame(ds.data, columns=COLUMNS[:4])
    # Restore UCI's original iris.data rows; no changes to the labels.
    df.loc[34, COLUMNS[:4]] = [4.9, 3.1, 1.5, 0.1]
    df.loc[37, COLUMNS[:4]] = [4.9, 3.1, 1.5, 0.1]
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    df["species"] = [labels[int(i)] for i in ds.target]
    out = ROOT / "data" / "iris_uci_original.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    assert len(df) == 150
    print(f"Prepared {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
