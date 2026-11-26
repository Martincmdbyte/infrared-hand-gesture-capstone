"""Compare random-frame and unseen-person evaluation on LeapGestRecog."""

import argparse
import csv
import hashlib
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.svm import LinearSVC

from features import edge_features, prepare_hand, read_gray

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GESTURES = [
    ("01_palm", "Palm"),
    ("02_l", "L_shaped"),
    ("03_fist", "Fist"),
    ("04_fist_moved", "Fist_moved"),
    ("05_thumb", "Thumb"),
    ("06_index", "Index"),
    ("07_ok", "Okay"),
    ("08_palm_moved", "Palm_moved"),
    ("09_c", "C_shape"),
    ("10_down", "Down"),
]
LABELS = [local.replace("_", " ") for original, local in GESTURES]
SEED = 42


def find_frames(data):
    frames = []
    person_folders = (data / "00").is_dir()
    for person in range(10):
        person_id = f"{person:02d}"
        for label, (original, local) in enumerate(GESTURES):
            if person_folders:
                paths = sorted((data / person_id / original).glob("*.png"))
            else:
                paths = sorted((data / local).glob(f"frame_{person_id}_{label + 1:02d}_*.png"))
            if len(paths) != 200:
                raise ValueError(
                    f"Expected 200 frames for person {person_id}, {LABELS[label]}. "
                    f"Found {len(paths)} under {data}. Check the data folder in the README."
                )
            for path in paths:
                frames.append((path, person_id, label))
    return frames


def remove_duplicate_frames(frames):
    by_image = {}
    for path, person, label in frames:
        gray = read_gray(path)
        digest = hashlib.sha256(gray.tobytes()).digest()
        by_image.setdefault((gray.size, digest), []).append((path, person, label))

    kept = []
    counts = {"source_frames": len(frames), "same_person_and_gesture_copies": 0,
              "shared_across_people_or_gestures": 0}
    for matching_frames in by_image.values():
        assignments = {(person, label) for path, person, label in matching_frames}
        if len(assignments) > 1:
            # No split can safely assign these images to a single person and gesture.
            counts["shared_across_people_or_gestures"] += len(matching_frames)
        else:
            kept.append(matching_frames[0])
            counts["same_person_and_gesture_copies"] += len(matching_frames) - 1

    counts["retained_frames"] = len(kept)
    return kept, counts


def load_features(frames):
    pixels = []
    edges = []
    labels = []
    people = []
    for path, person, label in frames:
        hand = prepare_hand(read_gray(path))
        pixels.append(hand.ravel())
        edges.append(edge_features(hand))
        labels.append(label)
        people.append(person)

    return {"Pixels": np.array(pixels), "Edges": np.array(edges)}, np.array(labels), np.array(people)


def fit_model(features, labels):
    model = LinearSVC(C=1.0, dual=False, tol=0.001, random_state=SEED, max_iter=5000)
    model.fit(features, labels)
    return model


def add_scores(scores, split, name, person, expected, predicted):
    scores.append({
        "split": split,
        "features": name,
        "held_out_person": person,
        "frames": len(expected),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, labels=np.unique(expected), average="macro"),
    })


def evaluate(features, labels, people):
    scores = []
    edge_predictions = np.empty_like(labels)
    indexes = np.arange(len(labels))
    train, test = train_test_split(indexes, test_size=0.1, stratify=labels, random_state=SEED)
    person_splits = list(LeaveOneGroupOut().split(indexes, labels, people))

    for name, values in features.items():
        model = fit_model(values[train], labels[train])
        guessed = model.predict(values[test])
        add_scores(scores, "random_frames", name, "", labels[test], guessed)

        predictions = np.empty_like(labels)
        for train_person, test_person in person_splits:
            person = people[test_person[0]]
            print(f"{name}: testing on person {person}", flush=True)
            model = fit_model(values[train_person], labels[train_person])
            predictions[test_person] = model.predict(values[test_person])
            add_scores(scores, "held_out_person", name, person, labels[test_person], predictions[test_person])
        add_scores(scores, "held_out_person_all", name, "", labels, predictions)
        if name == "Edges":
            edge_predictions = predictions

    return scores, confusion_matrix(labels, edge_predictions, labels=range(len(LABELS)))


def save_results(scores, confusion, audit, output):
    with (output / "data_audit.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit))
        writer.writeheader()
        writer.writerow(audit)

    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)

    with (output / "edge_confusion.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual / predicted", *LABELS])
        for label, counts in zip(LABELS, confusion):
            writer.writerow([label, *counts])

    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for name, color, marker in [("Pixels", "#ac5019", "s"), ("Edges", "#12665e", "o")]:
        rows = [row for row in scores if row["features"] == name and row["split"] == "held_out_person"]
        ax.plot(
            [row["held_out_person"] for row in rows],
            [row["accuracy"] * 100 for row in rows],
            label=name, color=color, marker=marker, linewidth=2,
        )
    ax.set(title="Accuracy varies across unseen people", xlabel="Person left out of training",
           ylabel="Accuracy (%)", ylim=(0, 104))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(output / "accuracy_by_person.png", dpi=120)
    plt.close(fig)

    percent = confusion / confusion.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.7, 6.3), layout="constrained")
    heatmap = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(heatmap, ax=ax, label="% of actual gesture")
    ax.set(title="Where the edge model makes mistakes on unseen people", xlabel="Predicted gesture",
           ylabel="Actual gesture", xticks=range(len(LABELS)), yticks=range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=40, ha="right")
    ax.set_yticklabels(LABELS)
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            value = percent[row, col]
            if value >= 0.5:
                color = "white" if value >= 50 else "#18334a"
                ax.text(col, row, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)
    fig.savefig(output / "edge_confusion.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/leapGestRecog"),
                        help="Directory with the gesture frames")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory for results and model")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    frames = find_frames(args.data)
    print(f"Checking {len(frames):,} frames for identical grayscale images.", flush=True)
    frames, audit = remove_duplicate_frames(frames)
    print(f"Extracting features from {len(frames):,} unique, consistently labeled frames.", flush=True)
    features, labels, people = load_features(frames)
    scores, confusion = evaluate(features, labels, people)
    save_results(scores, confusion, audit, args.output)

    # Fit a fresh edge model for trying the classifier after the study.
    model = fit_model(features["Edges"], labels)
    joblib.dump({"model": model, "labels": LABELS}, args.output / "edge_model.joblib")

    for row in scores:
        if row["split"] != "held_out_person":
            print(f'{row["features"]:6} | {row["split"]:19} | '
                  f'accuracy {row["accuracy"]:.1%} | macro F1 {row["macro_f1"]:.3f}')
    print(f"Saved study results and the prediction model in {args.output}.")


if __name__ == "__main__":
    main()
