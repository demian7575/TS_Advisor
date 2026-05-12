#!/usr/bin/env python
import argparse
from ts_advisor.inference import predict_one


def parse_args():
    p = argparse.ArgumentParser(description="Predict a design responsible MHO for a new TR-like input.")
    p.add_argument("--model", required=True, help="Path to saved joblib model bundle.")
    p.add_argument("--heading", default="")
    p.add_argument("--observation", default="")
    p.add_argument("--priority", default="")
    return p.parse_args()


def main():
    args = parse_args()
    result = predict_one(model_path=args.model, heading=args.heading, observation=args.observation, priority=args.priority)
    print(result)

if __name__ == "__main__":
    main()
