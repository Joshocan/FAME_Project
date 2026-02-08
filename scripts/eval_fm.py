#!/usr/bin/env python
"""CLI for FM coverage evaluation.

Example:
  PYTHONPATH=$(pwd) .venv/bin/python scripts/eval_fm.py \
      --gt data/ground_truth/federation.xml \
      --pred results/rag/ss-rgfm/fm/ss-rgfm_response_gpt-4.1_2026-02-08T12-19-44.xml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from fame.evaluation import CoverageEvaluator, CoverageConfig
from fame.utils.dirs import build_paths, ensure_for_stage


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate FM coverage (semantic recall)")
    ap.add_argument("--gt", required=True, help="Ground-truth FeatureIDE XML")
    ap.add_argument("--pred", required=True, help="Predicted/generated FeatureIDE XML")
    ap.add_argument("--model", default="all-mpnet-base-v2", help="SentenceTransformer model")
    ap.add_argument("--threshold", type=float, default=0.35, help="Similarity threshold")
    ap.add_argument("--top-k", type=int, default=3, help="Top-k matches to consider")
    ap.add_argument("--feature-weight", type=float, default=0.9, help="Weight for node similarity")
    ap.add_argument("--parent-weight", type=float, default=0.1, help="Weight for parent similarity")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-node output")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths()
    ensure_for_stage("evaluation", paths)

    cfg = CoverageConfig(
        model_name=args.model,
        similarity_threshold=args.threshold,
        top_k=args.top_k,
        feature_weight=args.feature_weight,
        parent_weight=args.parent_weight,
    )

    evaluator = CoverageEvaluator(cfg)
    gt_path = Path(args.gt)
    pred_path = Path(args.pred)
    score = evaluator.score(gt_path, pred_path, verbose=not args.quiet)

    if args.quiet:
        print(score)

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    out_name = f"coverage_{pred_path.stem}_vs_{gt_path.stem}_{timestamp}.json"
    out_file = paths.evaluation_coverage / out_name
    out = {
        "metric": "coverage_recall",
        "score": score,
        "ground_truth": str(gt_path),
        "prediction": str(pred_path),
        "model": args.model,
        "similarity_threshold": args.threshold,
        "top_k": args.top_k,
        "feature_weight": args.feature_weight,
        "parent_weight": args.parent_weight,
        "timestamp_utc": timestamp,
    }
    out_file.write_text(json.dumps(out, indent=2))
    if not args.quiet:
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
