"""Rank candidate biomarker genes in CuMiDa microarray datasets.

    python main.py data/Breast_GSE26910.csv [more.csv ...] \
        [--probe-map data/GPL570.txt] [--top-k 20] [--out results/]

For each dataset: repeated stratified cross-validation of two classifiers
(L1 logistic regression, random forest), fold-level metrics, and a ranking of
probes by how often they appear in a fold's top-k. With several datasets, the
probes ranked by all of them are reported as well.
"""
import argparse
import os

from biomarkers.data import load_cumida, load_probe_map
from biomarkers.selection import evaluate, make_classifiers, summarize_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="CuMiDa CSV files")
    parser.add_argument("--probe-map", help="GEO platform annotation table (probe -> gene)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--prefilter", type=int, default=2000,
                        help="keep the N most variable probes per training fold")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    probe_map = load_probe_map(args.probe_map) if args.probe_map else {}
    os.makedirs(args.out, exist_ok=True)
    stable = {}

    for path in args.datasets:
        name = os.path.splitext(os.path.basename(path))[0]
        X, y, classes = load_cumida(path)
        print(f"\n=== {name}: {X.shape[0]} samples ({(y == 0).sum()} {classes[0]}, "
              f"{(y == 1).sum()} {classes[1]}), {X.shape[1]} probes")

        for clf_name, clf in make_classifiers(args.seed).items():
            result = evaluate(X, y, clf, n_splits=args.splits, n_repeats=args.repeats,
                              top_k=args.top_k, prefilter=args.prefilter, seed=args.seed)
            ranking = result.ranking
            if probe_map:
                ranking.insert(0, "gene", [probe_map.get(p, "") for p in ranking.index])
            ranking.to_csv(os.path.join(args.out, f"{name}_{clf_name}_ranking.csv"),
                           index_label="probe")
            result.metrics.to_csv(os.path.join(args.out, f"{name}_{clf_name}_folds.csv"),
                                  index=False)

            print(f"\n[{clf_name}]")
            print(summarize_metrics(result.metrics).round(3).to_string())
            print(ranking.head(10).round(3).to_string())
            stable.setdefault(clf_name, []).append(
                set(ranking.index[ranking["selection_frequency"] >= 0.5]))

    if len(args.datasets) > 1:
        for clf_name, sets in stable.items():
            shared = set.intersection(*sets)
            print(f"\n[{clf_name}] probes selected in >=50% of folds in every dataset: "
                  f"{sorted(shared) or 'none'}")


if __name__ == "__main__":
    main()
