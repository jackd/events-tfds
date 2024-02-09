import typing as tp

import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

import events_tfds.events.vdt_davis  # pylint:disable=unused-import


def get_splits(key_time, query_time):
    splits = np.empty((key_time.shape[0] + 1,), dtype=np.int64)
    splits[0] = 0
    index = 0
    for i, t in enumerate(key_time):
        while index < len(query_time) and query_time[index] < t:
            index += 1
        if index == len(query_time):
            splits[i + 1 :] = index
            break
        splits[i + 1] = index
    return splits


def animate_frames(
    images: np.ndarray,
    image_times: np.ndarray,
    event_coords: np.ndarray,
    event_polarity: np.ndarray,
    event_times: np.ndarray,
    bboxes,
    label_ids: np.ndarray,
    label_times: np.ndarray,
    fps: int,
    save_file: str = None,
):
    """
    Creates an animation of a sequence of frames.

    Args:
        frames: list of frame indices to animate
        fps: frames per second for animation
        save_file: optional file name to save the animation as an mp4
    """
    fig, ax = plt.subplots()
    label_splits = get_splits(image_times, label_times)
    event_splits = get_splits(image_times, event_times)

    def update(frame):
        ax.clear()
        label_start, label_end = label_splits[frame : frame + 2]
        event_start, event_end = event_splits[frame : frame + 2]
        add_to_axes(
            ax,
            images[frame],
            event_coords[event_start:event_end],
            event_polarity[event_start:event_end],
            bboxes[label_start:label_end],
            label_ids[label_start:label_end],
        )

    anim = FuncAnimation(fig, update, frames=range(len(images)), interval=1000 // fps)

    if save_file:
        anim.save(save_file, fps=fps, dpi=100)
    else:
        plt.show()


def add_bounding_box(
    ax: plt.Axes, ymin: int, xmin: int, ymax: int, xmax: int, label: str, color="r"
):
    """
    Args:
        ax: axes with image pre-plotted
        ymin, xmin, ymax, xmax: bounding box parameters
        label: string label to display
    """
    # Calculate the width and height of the bounding box
    width = xmax - xmin
    height = ymax - ymin

    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle(
        (xmin, ymin), width, height, linewidth=2, edgecolor=color, facecolor="none"
    )

    # Add the rectangle patch to the axes
    ax.add_patch(rect)

    # Add the label to the axes
    ax.text(xmin, ymin, label, fontsize=10, color=color)


RED = (255, 0, 0)
BLUE = (0, 0, 255)


def add_to_axes(
    ax: plt.Axes,
    image: np.ndarray,
    coords: np.ndarray,
    polarity: np.ndarray,
    boxes: tp.Tuple[tfds.features.BBox],
    object_ids: tp.Tuple[int],
):
    image = np.tile(image, (1, 1, 3))
    x, y = coords.T
    image[y[polarity], x[polarity]] = RED
    image[y[~polarity], x[~polarity]] = BLUE
    y_scale, x_scale = image.shape[:2]
    plt.imshow(image)
    for box, object_id in zip(boxes, object_ids):
        ymin, xmin, ymax, xmax = box
        add_bounding_box(
            ax,
            int(ymin * y_scale),
            int(xmin * x_scale),
            int(ymax * y_scale),
            int(xmax * x_scale),
            f"object_{object_id}",
            color="white",
        )


def main():
    fps = 384
    dataset = tfds.load(f"vdt_davis/{fps}hz", split="Scene_A", as_supervised=True)
    repeats = 384 // 24

    for (frames, events), labels in dataset:
        images = frames["image"].numpy()
        images = np.tile(np.expand_dims(images, axis=1), (1, repeats, 1, 1, 1))
        images = images.reshape(-1, *images.shape[2:])
        image_times = frames["time"].numpy()
        image_times = np.expand_dims(image_times, 1) + (
            np.arange(0, repeats) / fps * 1e9
        ).astype(np.int64)
        image_times = image_times.reshape(-1)

        event_time = events["time"].numpy()
        event_coords = events["coords"].numpy()
        event_polarity = events["polarity"].numpy()
        bboxes = labels["bounding_box"].numpy()
        object_ids = labels["object_id"].numpy()
        label_times = labels["time"].numpy()
        animate_frames(
            images,
            image_times,
            event_coords,
            event_polarity,
            event_time,
            bboxes,
            object_ids,
            label_times,
            fps=24,  # always visualize at 24 fps
        )


if __name__ == "__main__":
    main()
