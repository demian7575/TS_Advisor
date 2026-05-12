#!/usr/bin/env python
import argparse
from ts_advisor.config import TSAdvisorConfig
from ts_advisor.logging import ProgressLogger
from ts_advisor.pipeline import run_training_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Train the TS Advisor baseline model comparison pipeline.")
    p.add_argument("--csv", required=True, help="Path to 4G/5G Trouble Report CSV.")
    p.add_argument("--run-mode", default="fast", choices=["fast", "balanced", "full"])
    p.add_argument("--model-out", default="models/ts_advisor_model.joblib")
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--drop-other", action="store_true")
    p.add_argument("--split-mode", default="random_stratified", choices=["random_stratified", "temporal"])
    return p.parse_args()


def main():
    args = parse_args()
    config = TSAdvisorConfig(run_mode=args.run_mode, model_path=args.model_out, min_samples=args.min_samples, drop_other=args.drop_other, split_mode=args.split_mode)
    outputs = run_training_pipeline(args.csv, config=config, logger=ProgressLogger())
    cols = ["model", "feature_set", "train_accuracy", "val_accuracy", "macro_f1", "weighted_f1", "time_s", "warnings"]
    print("\nBaseline comparison:")
    print(outputs["results_df"][[c for c in cols if c in outputs["results_df"].columns]].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
