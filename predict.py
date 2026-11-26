"""Predict the gesture in one Leap Motion infrared image."""

import argparse
from pathlib import Path

import joblib

from features import edge_features, prepare_hand, read_gray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("outputs/edge_model.joblib"))
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"{args.model} is missing. Run study.py first.")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    saved = joblib.load(args.model)
    hand = prepare_hand(read_gray(args.image))
    features = edge_features(hand).reshape(1, -1)
    label = saved["model"].predict(features)[0]
    print(f"Predicted gesture: {saved['labels'][label]}")


if __name__ == "__main__":
    main()
