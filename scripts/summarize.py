#!/usr/bin/env python
"""Aggregate evaluated runs into one comparison table (category means + system metrics).

    python scripts/summarize.py --runs-root runs/mvtec_ad --tag test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from smartlens.config import load_yaml, resolve_path  # noqa: E402
from smartlens.evaluation.evaluator import results_table  # noqa: E402

COLS = ["P-AP", "AU-PRO@5%FPR", "AU-PRO", "I-FPR@95TPR", "I-TPR@1%FPR",
        "I-AUROC", "I-AP", "I-F1max", "at_tau_unsup/precision", "at_tau_unsup/recall",
        "at_tau_unsup/fpr", "at_tau_f1/precision", "at_tau_f1/recall", "at_tau_f1/f1", "at_tau_f1/fpr",
        "P-AUROC", "P-F1max"]
SYS = ["latency_ms_mean", "fps", "peak_vram_MB", "params_total_M", "params_trainable_M"]
ORDER = ["ssim_cae", "patchcore", "dinomaly"]


def comparison_protocol(data: dict, cfg: dict) -> dict:
    """Use result provenance where available; identify legacy results explicitly."""
    p = data.get("protocol", {})
    return {
        "dataset": p.get("dataset", cfg.get("data", {}).get("name", "unknown")),
        "samples": sorted((r["category"], r["split"], r["clean"]["n_images"], r["clean"]["n_anomalous"])
                          for r in data["results"]),
        "mask_size": p.get("mask_size", cfg.get("data", {}).get("mask_size", "unknown")),
        "mask_resize": p.get("mask_resize", cfg.get("data", {}).get("mask_resize", "area")),
        "manifest_sha256": p.get("manifest_sha256", "legacy-unrecorded"),
        "pro_integration_limit": p.get("pro_integration_limit", cfg.get("eval", {}).get("pro_integration_limit", 0.3)),
        "unsup_percentile": p.get("unsup_percentile", cfg.get("eval", {}).get("unsup_percentile", 99.0)),
        "calibration_scope": p.get("calibration_scope", "per_category"),
        "perturbations": sorted({v for r in data["results"] for v in r.get("optshift", {})}),
    }


def summarize(runs_root: Path, tag: str) -> pd.DataFrame:
    rows, per_cat, coverage = {}, {}, {}
    reference = None
    for f in sorted(runs_root.glob(f"*/eval/results_{tag}.json")):
        run = f.parent.parent.name
        data = json.loads(f.read_text())
        cfg_path = f.parent.parent / "config.yaml"
        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        name = data.get("protocol", {}).get("model", cfg.get("model", {}).get("name", run))
        if name in ("ensemble", "efficientad"):  # retired models; historical artifacts are not active comparisons
            continue
        protocol = comparison_protocol(data, cfg)
        if reference is not None and protocol != reference:
            different = [k for k in protocol if protocol[k] != reference[k]]
            raise ValueError(f"Incomparable evaluation protocols in {run}: {different}. "
                             "Use identical datasets, categories, splits and evaluation settings.")
        reference = protocol
        df = results_table(data["results"])
        row = {c: df.loc["mean", c] for c in COLS if c in df.columns}
        row.update({k: data.get("latency", {}).get(k) for k in SYS})
        row["n_categories"] = len(data["results"])
        row["n_images"] = sum(r["clean"]["n_images"] for r in data["results"])
        configured = data.get("protocol", {}).get("configured_categories", cfg.get("data", {}).get("categories", []))
        evaluated = sorted(r["category"] for r in data["results"])
        coverage[run] = (evaluated, configured, bool(data.get("protocol")))
        rows[run] = row
        per_cat[run] = df.drop(index="mean")["I-AUROC"]
        for p in sorted({p for r in data["results"] for p in r.get("optshift", {})}):
            dfp = results_table(data["results"], pert=p)
            for metric in ("I-AUROC", "P-AP", "AU-PRO", "AU-PRO@5%FPR", "at_tau_unsup/recall", "at_tau_unsup/fpr"):
                if metric in dfp:
                    row[f"optshift/{p}/{metric}"] = dfp.loc["mean", metric]
    if not rows:
        raise SystemExit(f"No results_{tag}.json found under {runs_root}/*/eval/")
    order = [r for r in ORDER if r in rows] + sorted(r for r in rows if r not in ORDER)
    table = pd.DataFrame(rows).T.loc[order]
    table.to_csv(runs_root / f"comparison_{tag}.csv", float_format="%.5f")
    pd.DataFrame(per_cat)[order].to_csv(runs_root / f"comparison_{tag}_per_category_I-AUROC.csv", float_format="%.5f")

    pct = [c for c in table.columns if c not in SYS and c not in ("n_categories", "n_images")]
    lines = [f"# Baseline comparison — {runs_root.name} ({tag})", "",
             "Metrics in % (macro mean over the listed evaluated categories). `unsup` = threshold at the 99th pct of "
             "normal-validation scores; `f1tuned` = F1-optimal threshold on the stratified test_val split. "
             "F1max and TPR@1%FPR are test-curve diagnostics.", "",
             f"Dataset: **{reference['dataset']}**; mask grid: **{reference['mask_size']}** "
             f"({reference['mask_resize']}).", "",
             "## Coverage", "", "| model | evaluated/configured categories | categories | provenance |", "|---|---|---|---|"]
    for run, (cats, configured, recorded) in coverage.items():
        status = "complete" if configured and set(cats) == set(configured) else "PARTIAL/UNVERIFIED"
        lines.append(f"| {run} | {len(cats)}/{len(configured) or '?'} ({status}) | {', '.join(cats)} | "
                     f"{'recorded' if recorded else 'legacy; training coverage unverified'} |")
    lines += ["", "Partial-category and shortened-training runs are smoke tests, not evidence of SOTA performance.", ""]
    head = ["model", "categories", "test images"] + [c.replace("at_tau_unsup/", "unsup-").replace("at_tau_f1/", "f1tuned-") for c in pct] + \
        ["ms/img", "FPS", "VRAM MB", "Params M"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|---" * len(head) + "|")
    for run, r in table.iterrows():
        cells = ["–" if pd.isna(r[c]) else f"{100 * float(r[c]):.1f}" for c in pct]
        sysc = [r.get("latency_ms_mean"), r.get("fps"), r.get("peak_vram_MB"), r.get("params_total_M")]
        cells += ["–" if v is None or pd.isna(v) else f"{float(v):.1f}" for v in sysc]
        lines.append(f"| {run} | {int(r['n_categories'])} | {int(r['n_images'])} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-category I-AUROC (%)", "",
              "| category | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
    pc = pd.DataFrame(per_cat)[order]
    for cat, r in pc.iterrows():
        lines.append(f"| {cat} | " + " | ".join("–" if pd.isna(v) else f"{100 * v:.1f}" for v in r) + " |")
    (runs_root / f"comparison_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs/mvtec_ad")
    ap.add_argument("--tag", default="test")
    args = ap.parse_args()
    summarize(resolve_path(args.runs_root), args.tag)


if __name__ == "__main__":
    main()
