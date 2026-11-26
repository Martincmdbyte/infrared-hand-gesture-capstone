"""Prepare one infrared frame for the gesture classifiers."""

import numpy as np
from PIL import Image


def read_gray(path):
    with Image.open(path) as image:
        return image.convert("L")


def prepare_hand(gray):
    pixels = np.asarray(gray)

    # The Leap Motion frames have a bright hand on a dark background.
    rows, columns = np.where(pixels >= 80)
    if len(rows):
        padding = 12
        left = max(0, columns.min() - padding)
        top = max(0, rows.min() - padding)
        right = min(gray.width, columns.max() + padding + 1)
        bottom = min(gray.height, rows.max() + padding + 1)
        gray = gray.crop((left, top, right, bottom))

    # Keep the hand's proportions when fitting it into a square.
    hand = gray.copy()
    hand.thumbnail((40, 40), Image.Resampling.BILINEAR)
    square = Image.new("L", (40, 40))
    offset = ((40 - hand.width) // 2, (40 - hand.height) // 2)
    square.paste(hand, offset)
    return np.asarray(square, dtype=np.float32) / 255


def edge_features(hand):
    # Measure the direction and strength of the brightness changes.
    dy, dx = np.gradient(hand)
    strength = np.hypot(dx, dy)
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    direction = (angle / 20).astype(int).clip(0, 8)

    # Summarize nine edge directions in each 8-by-8 patch.
    features = []
    for row in range(0, 40, 8):
        for col in range(0, 40, 8):
            directions = direction[row:row + 8, col:col + 8].ravel()
            weights = strength[row:row + 8, col:col + 8].ravel()
            histogram = np.bincount(directions, weights=weights, minlength=9)
            features.extend(histogram / (np.linalg.norm(histogram) + 1e-6))

    return np.array(features, dtype=np.float32)
