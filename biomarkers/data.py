"""Loading CuMiDa microarray CSVs and (optionally) mapping probes to genes."""
import numpy as np
import pandas as pd


def load_cumida(path):
    """Read a CuMiDa CSV (columns: samples, type, <probe ids>...).

    Returns X (samples x probes DataFrame), y (0/1 array), and the class names,
    where the positive class (1) is any label other than 'normal'.
    """
    df = pd.read_csv(path)
    if "type" not in df.columns:
        raise ValueError(f"{path}: expected a 'type' column")
    X = df.drop(columns=[c for c in ("samples", "type") if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    labels = df["type"].astype(str)
    negative = "normal" if "normal" in set(labels) else sorted(labels.unique())[0]
    y = (labels != negative).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError(f"{path}: expected exactly two classes, got {sorted(labels.unique())}")
    positive = sorted(set(labels) - {negative})
    return X, y, [negative, "/".join(positive)]


def load_probe_map(path, probe_col="ID", gene_col="Gene Symbol"):
    """Read a GEO platform annotation table (GPL*.txt / .csv / .xlsx) as {probe: gene}."""
    if str(path).endswith((".xls", ".xlsx")):
        table = pd.read_excel(path)
    else:
        table = pd.read_csv(path, sep=None, engine="python", comment="#")
    table = table[[probe_col, gene_col]].dropna()
    # Some probes map to several genes ("A /// B"); keep the first symbol.
    table[gene_col] = table[gene_col].astype(str).str.split("///").str[0].str.strip()
    return dict(zip(table[probe_col].astype(str), table[gene_col]))
