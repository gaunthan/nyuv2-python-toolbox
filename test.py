#!/usr/bin/env python3

from pathlib import Path
from nyuv2 import *
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = Path('dataset')

def plot_color(ax, color, title="Color"):
    """Displays a color image from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(color)

def plot_depth(ax, depth, title="Depth"):
    """Displays a depth map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(depth, cmap='Spectral')

def plot_label(ax, label, title="Label"):
    """Displays a label map from the NYU dataset."""

    ax.axis('off')
    ax.set_title(title)
    ax.imshow(label)

def test_labeled_dataset():
    image_index = 0

    labeled = LabeledDataset(DATASET_DIR / 'nyu_depth_v2_labeled.mat')
    color, depth, label, scene_type, scene_name = labeled[image_index]

    fig = plt.figure('No. %d %s, %s' % (image_index, scene_type, scene_name), figsize=(12, 5))

    ax = fig.add_subplot(1, 3, 1)
    plot_color(ax, color)

    ax = fig.add_subplot(1, 3, 2)
    plot_depth(ax, depth)

    ax = fig.add_subplot(1, 3, 3)
    plot_label(ax, label)

    plt.show()

    labeled.close()

def test_raw_dataset():
    # Pick the first raw dataset part we find
    try:
        raw_archive_path = next(DATASET_DIR.glob('*.zip'))
    except:
        return

    raw_archive = RawDatasetArchive(raw_archive_path)
    frame = raw_archive[5]
    depth_path, color_path = Path('.') / frame[0], Path('.') / frame[1]

    if not (depth_path.exists() and color_path.exists()):
        raw_archive.extract_frame(frame)

    color = load_color_image(color_path)
    depth = load_depth_image(depth_path)

    fig = plt.figure("Raw Dataset Sample", figsize=(12, 5))

    before_proj_overlay = color_depth_overlay(color, depth, relative=True)

    ax = fig.add_subplot(1, 2, 1)
    plot_color(ax, before_proj_overlay, "Before Projection")

    # TODO: project depth and RGB image

    plt.show()

test_labeled_dataset()
test_raw_dataset()
