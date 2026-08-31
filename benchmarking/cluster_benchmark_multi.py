from pathlib import Path
from typing import List, Optional

import pandas as pd
import scanpy as sc
from cluster_eval_with_ref import compute_ARI, compute_NMI, filter_matched_labels
from cluster_eval_without_ref import compute_ASW, compute_CHAOS, compute_PAS
from tqdm.contrib.concurrent import process_map


def benchmark_single_cluster_sample(args):
    """
    Evaluate a single clustering result using metrics from cluster_eval_without_ref.py
    args: [name, input_path, pred_key, sample_size, reference_path, gt_key, metrics]
    """
    name, input_path, pred_key, sample_size, reference_path, gt_key, metrics = args
    adata = sc.read_h5ad(input_path)
    result = {"name": name}

    # Calculate metrics based on the metrics list
    if "CHAOS" in metrics:
        chaos = compute_CHAOS(adata, pred_key)
        result["CHAOS(down)"] = chaos

    if "PAS" in metrics:
        pas = compute_PAS(adata, pred_key, spatial_key="spatial")
        result["PAS(down)"] = pas

    if "ASW" in metrics:
        asw = compute_ASW(adata, pred_key, spatial_key="spatial", sample_size=sample_size)
        result["ASW(up)"] = asw

    # If only gt_key is provided, use input as reference
    if (not reference_path or str(reference_path) == "nan") and gt_key and str(gt_key) != "nan":
        # Filter NaN labels using the same adata for both reference and prediction
        pred_labels_valid, gt_labels_valid = filter_matched_labels(adata, adata, gt_key, pred_key)
        if "ARI" in metrics:
            result["ARI(up)"] = compute_ARI(pred_labels_valid, gt_labels_valid)
        if "NMI" in metrics:
            result["NMI(up)"] = compute_NMI(pred_labels_valid, gt_labels_valid)
    # If reference and gt_key are provided, compute accuracy metrics
    elif reference_path and gt_key and str(reference_path) != "nan" and str(gt_key) != "nan":
        ref_adata = sc.read_h5ad(reference_path)
        # Filter and align labels using the shared function
        pred_labels_valid, gt_labels_valid = filter_matched_labels(ref_adata, adata, gt_key, pred_key)
        if "ARI" in metrics:
            result["ARI(up)"] = compute_ARI(pred_labels_valid, gt_labels_valid)
        if "NMI" in metrics:
            result["NMI(up)"] = compute_NMI(pred_labels_valid, gt_labels_valid)
    return result


def benchmark_multi(
    input_df: pd.DataFrame, max_workers: int = 1, sample_size: int = 100000, metrics: Optional[List[str]] = None
) -> List[dict]:
    """
    Evaluate multiple clustering results in parallel.
    input_df: DataFrame with columns: name, input, pred_key, marker_key, cluster_used_key, reference_path, gt_key
    metrics: List of metrics to calculate. If None, all metrics will be calculated.
    """
    if metrics is None:
        metrics = [
            "CHAOS",
            "PAS",
            "ASW",
            "ARI",
            "NMI",
        ]

    benchmark_args = [
        [
            row["name"],
            row["input"],
            row["pred_key"],
            sample_size,
            row.get("reference_path"),
            row.get("gt_key"),
            metrics,
        ]
        for i, row in list(input_df.iterrows())
    ]
    results = process_map(benchmark_single_cluster_sample, benchmark_args, max_workers=max_workers)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate multiple clustering results at once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input csv file with columns: name, input, pred_key, reference_path, gt_key.",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory containing result figures and summary table"
    )
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers to run in parallel")
    parser.add_argument("--sample_size", type=int, default=100000, help="Sample size for silhouette score (ASW)")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metrics to calculate. Available metrics: CHAOS, PAS, ASW, ARI, NMI. If not specified, all metrics will be calculated.",
    )
    args = parser.parse_args()

    input_df = pd.read_csv(args.input, comment="#")
    print(input_df)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    results = benchmark_multi(input_df, max_workers=args.n_workers, sample_size=args.sample_size, metrics=args.metrics)
    result_df = pd.DataFrame(results)
    result_df.to_csv(Path(args.output).joinpath("cluster_benchmarks.csv"), index=False)
    print("Benchmark results saved to:", Path(args.output).joinpath("cluster_benchmarks.csv"))
