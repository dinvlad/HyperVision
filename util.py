from argparse import ArgumentTypeError
from threading import Thread
from typing import Optional, Tuple

import cv2
import numpy as np
from screeninfo import Monitor


def arg_pos_int(s: str):
    try:
        v = int(s)
        if v <= 0:
            raise ValueError()
        return v
    except ValueError:
        raise ArgumentTypeError("has to be a positive integer")


def arg_non_neg_int(s: str):
    try:
        v = int(s)
        if v < 0:
            raise ValueError()
        return v
    except ValueError:
        raise ArgumentTypeError("has to be a non-negative integer")


def arg_pos_float(s: str):
    try:
        value = float(s)
        if value <= 0.0:
            raise ValueError()
        return value
    except ValueError:
        raise ArgumentTypeError("has to be a positive float")


def arg_scale_float(s: str):
    try:
        v = float(s)
        if v <= 0.0 or v > 1.0:
            raise ValueError()
        return v
    except ValueError:
        raise ArgumentTypeError("has to be a number in the range (0.0, 1.0]")


class VideoCapture:
    """Implements threaded OpenCV video capture"""

    def __init__(self, idx: int, res: Tuple[int, int], fps: Optional[int] = None):
        # initialize the camera and start streaming
        self._cap = cv2.VideoCapture(idx)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[0])
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[1])
        if fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
        self._frame = None
        self._start()

    def __del__(self):
        self._cap.release()

    def _start(self):
        Thread(target=self._update, name="VideoCapture", daemon=True).start()

    def _update(self):
        retval = True
        while retval:
            retval, self._frame = self._cap.read()

    def read(self):
        frame = self._frame
        self._frame = None
        return frame


def imshow(
    win_name: str,
    img: np.array,
    pos: Tuple[int, int],
    size: Tuple[int, int],
    text="",
    shown=False,
):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    if not shown:
        cv2.moveWindow(win_name, pos[1], pos[0])
        cv2.resizeWindow(win_name, size[1], size[0])

    if text:
        cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(win_name, img)
    return True


def show_preview(
    cap: VideoCapture,
    mon: Monitor,
    scale: float,
    frame_delay_ms: int,
):
    win_name = "Preview"

    width = round(mon.width * scale)
    x0 = round((mon.width - width) / 2)

    shown = False
    key = -1

    while key == -1:
        img = cap.read()
        if img is None:
            continue

        height = round(img.shape[0] * width / img.shape[1])
        y0 = round((mon.height - height) / 2)

        shown = imshow(
            win_name,
            img,
            (y0, x0),
            (height, width),
            "Move the screen into the field of view and press any key",
            shown,
        )

        key = cv2.waitKey(frame_delay_ms)

    cv2.destroyWindow(win_name)
