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
from smartlens.evaluation.evaluator import flatten_result, results_table  # noqa: E402
from smartlens.evaluation.metrics import paired_bootstrap_ci  # noqa: E402

COLS = ["P-AP", "Img-P-AP", "AU-PRO@5%FPR", "AU-PRO", "I-FPR@95TPR", "I-TPR@1%FPR",
        "I-AUROC", "I-AP", "I-F1max", "at_tau_unsup/precision", "at_tau_unsup/recall",
        "at_tau_unsup/fpr", "at_tau_f1/precision", "at_tau_f1/recall", "at_tau_f1/f1", "at_tau_f1/fpr",
        "P-AUROC", "P-F1max"]
SYS = ["latency_ms_mean", "latency_ms_max_category", "fps", "peak_vram_MB",
       "params_total_M", "params_trainable_M", "buffers_M", "deployed_checkpoints", "deployed_footprint_M"]
ORDER = ["ssim_cae", "patchcore", "patchcore_448", "dinomaly"]


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
        "amp": p["amp"] if "amp" in p else "legacy-unrecorded",
        "allow_tf32": p["allow_tf32"] if "allow_tf32" in p else "legacy-unrecorded",
        "perturbations": sorted({v for r in data["results"] for v in r.get("optshift", {})}),
    }


def _fmt_sys(val, is_int: bool = False) -> str:
    if val is None or pd.isna(val):
        return "–"
    return f"{int(round(float(val)))}" if is_int else f"{float(val):.1f}"


def summarize(runs_root: Path, tag: str, n_boot: int = 1000) -> pd.DataFrame:
    rows, per_cat, coverage, raw_results, pooled_rows = {}, {}, {}, {}, {}
    reference = None
    for f in sorted(runs_root.glob(f"*/eval/results_{tag}.json")):
        run = f.parent.parent.name
        data = json.loads(f.read_text())
        cfg_path = f.parent.parent / "config.yaml"
        cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
        p_meta = data.get("protocol", {})
        arch = p_meta.get("model_arch", p_meta.get("model", cfg.get("model", {}).get("name", run)))
        if arch in ("ensemble", "efficientad") or run in ("ensemble", "efficientad"):  # retired historical artifacts
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
        configured = p_meta.get("configured_categories", cfg.get("data", {}).get("categories", []))
        evaluated = sorted(r["category"] for r in data["results"])
        coverage[run] = (evaluated, configured, bool(p_meta))
        rows[run] = row
        raw_results[run] = data["results"]
        per_cat[run] = df.drop(index="mean")["I-AUROC"]
        if "pooled_clean" in data:
            pooled_rows[run] = flatten_result(data["pooled_clean"])
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

    prec = ("legacy-unrecorded" if reference["amp"] == "legacy-unrecorded" else
            ("BF16 AMP" if reference["amp"] else ("FP32 (TF32)" if reference["allow_tf32"] else "FP32 (IEEE)")))
    pct = [c for c in table.columns if c not in SYS and c not in ("n_categories", "n_images")]
    lines = [f"# Baseline comparison — {runs_root.name} ({tag})", "",
             "Metrics in % (macro mean over the listed evaluated categories). `unsup` = threshold at the 99th pct of "
             "normal-validation scores; `f1tuned` = F1-optimal threshold on the stratified test_val split. "
             "F1max and TPR@1%FPR are test-curve diagnostics.", "",
             f"Dataset: **{reference['dataset']}**; mask grid: **{reference['mask_size']}** "
             f"({reference['mask_resize']}); calibration: **{reference['calibration_scope']}**; "
             f"precision: **{prec}**.", "",
             "## Coverage", "", "| model | evaluated/configured categories | categories | provenance |", "|---|---|---|---|"]
    for run, (cats, configured, recorded) in coverage.items():
        status = "complete" if configured and set(cats) == set(configured) else "PARTIAL/UNVERIFIED"
        lines.append(f"| {run} | {len(cats)}/{len(configured) or '?'} ({status}) | {', '.join(cats)} | "
                     f"{'recorded' if recorded else 'legacy; training coverage unverified'} |")
    lines += ["", "Partial-category and shortened-training runs are smoke tests, not evidence of SOTA performance.", ""]
    head = ["model", "categories", "test images"] + \
        [c.replace("at_tau_unsup/", "unsup-").replace("at_tau_f1/", "f1tuned-") for c in pct] + \
        ["ms/img", "max ms/img", "FPS", "VRAM MB", "Params M", "Deployed M", "Ckpts"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|---" * len(head) + "|")
    for run, r in table.iterrows():
        cells = ["–" if pd.isna(r[c]) else f"{100 * float(r[c]):.1f}" for c in pct]
        cells += [
            _fmt_sys(r.get("latency_ms_mean")),
            _fmt_sys(r.get("latency_ms_max_category")),
            _fmt_sys(r.get("fps")),
            _fmt_sys(r.get("peak_vram_MB")),
            _fmt_sys(r.get("params_total_M")),
            _fmt_sys(r.get("deployed_footprint_M")),
            _fmt_sys(r.get("deployed_checkpoints"), is_int=True),
        ]
        lines.append(f"| {run} | {int(r['n_categories'])} | {int(r['n_images'])} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-category I-AUROC (%)", "",
              "| category | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
    pc = pd.DataFrame(per_cat)[order]
    for cat, r in pc.iterrows():
        lines.append(f"| {cat} | " + " | ".join("–" if pd.isna(v) else f"{100 * v:.1f}" for v in r) + " |")

    if pooled_rows:
        p_order = [r for r in order if r in pooled_rows]
        pooled_df = pd.DataFrame(pooled_rows).T.loc[p_order]
        pooled_df.to_csv(runs_root / f"comparison_{tag}_pooled.csv", float_format="%.5f")
        p_cols = [c for c in ("I-AUROC", "I-AP", "I-FPR@95TPR", "I-TPR@1%FPR",
                              "at_tau_unsup/precision", "at_tau_unsup/recall", "at_tau_unsup/fpr",
                              "at_tau_f1/f1", "at_tau_f1/fpr") if c in pooled_df.columns]
        p_head = ["model"] + [c.replace("at_tau_unsup/", "unsup-").replace("at_tau_f1/", "f1tuned-") for c in p_cols]
        lines += ["", "## Category-blind pooled image ranking (%)", "",
                  "| " + " | ".join(p_head) + " |", "|---" * len(p_head) + "|"]
        for run, r in pooled_df.iterrows():
            cells = ["–" if pd.isna(r[c]) else f"{100 * float(r[c]):.1f}" for c in p_cols]
            lines.append(f"| {run} | " + " | ".join(cells) + " |")

    boot_ready = [r for r in order if all(bool(c.get("predictions")) for c in raw_results[r])]
    if len(boot_ready) >= 2 and n_boot > 0:
        ref_name = "dinomaly" if "dinomaly" in boot_ready else boot_ready[0]
        boot_rows = []
        for cand_name in boot_ready:
            if cand_name == ref_name:
                continue
            for item in paired_bootstrap_ci(raw_results[ref_name], raw_results[cand_name], n_boot=n_boot):
                boot_rows.append({"reference": ref_name, "candidate": cand_name, **item})
        if boot_rows:
            boot_df = pd.DataFrame(boot_rows)
            boot_df.to_csv(runs_root / f"comparison_{tag}_paired_bootstrap.csv", index=False, float_format="%.5f")
            lines += ["", f"## Paired image-level bootstrap (95% CI vs. `{ref_name}`, {n_boot} resamples)", "",
                      "| candidate | metric | ref (%) | cand (%) | diff (%pt) | 95% CI (%pt) | excludes 0 |",
                      "|---|---|---:|---:|---:|---:|---|"]
            for _, br in boot_df.iterrows():
                m_short = str(br["metric"]).replace("at_tau_unsup/", "unsup-").replace("at_tau_f1/", "f1tuned-")
                ci_str = (f"[{100 * br['ci_low']:+.2f}, {100 * br['ci_high']:+.2f}]"
                          if pd.notna(br["ci_low"]) and pd.notna(br["ci_high"]) else "–")
                lines.append(
                    f"| {br['candidate']} | {m_short} | {100 * br['ref']:.2f} | {100 * br['cand']:.2f} | "
                    f"{100 * br['diff']:+.2f} | {ci_str} | {'yes' if br['ci_excludes_zero'] else 'no'} |"
                )

    (runs_root / f"comparison_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs/mvtec_ad")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    args = ap.parse_args()
    summarize(resolve_path(args.runs_root), args.tag, n_boot=args.bootstrap_iters)


if __name__ == "__main__":
    main()
