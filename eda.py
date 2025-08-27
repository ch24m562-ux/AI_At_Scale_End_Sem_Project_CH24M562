import sys, json, pandas as pd
from pathlib import Path

inp, outdir = sys.argv[1], Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp)

summary = {
    "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "na_counts": df.isna().sum().to_dict(),
    "na_ratio": (df.isna().mean().round(4)).to_dict(),
    "nunique": df.nunique(dropna=True).to_dict(),
}
# target distribution if present
for target in ["Survived", "target", "label"]:
    if target in df.columns:
        vc = df[target].value_counts(dropna=False, normalize=False).to_dict()
        vr = df[target].value_counts(normalize=True, dropna=False).round(4).to_dict()
        summary["target"] = {"name": target, "counts": vc, "ratio": vr}
        break

# numeric summary (compact)
num_cols = df.select_dtypes(include="number")
if not num_cols.empty:
    desc = num_cols.describe().T.round(4)
    summary["numeric_describe"] = desc.to_dict(orient="index")
    # quick correlations (optional, clipped to 20x20)
    corr = num_cols.corr(numeric_only=True)
    if corr.shape[0] <= 20:
        summary["corr"] = corr.round(4).to_dict(orient="split")

(Path(outdir) / "summary.json").write_text(json.dumps(summary, indent=2))
print("Wrote:", (Path(outdir) / "summary.json"))
