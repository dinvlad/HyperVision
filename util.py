#!/usr/bin/env python3

import atexit
import json
from argparse import ArgumentTypeError
from dataclasses import dataclass
from enum import Enum
from threading import Thread
from time import sleep
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
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
        if v < 0.0 or v > 1.0:
            raise ValueError()
        return v
    except ValueError:
        raise ArgumentTypeError("has to be a number in the range [0.0, 1.0]")


class CameraModel(str, Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"

    def __str__(self):
        return self.value


@dataclass
class CameraParams:
    model: CameraModel
    dims: Tuple[int, int]
    k: np.array
    d: np.array

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(
                {
                    "model": self.model,
                    "dims": self.dims,
                    "k": self.k.tolist(),
                    "d": self.d.tolist(),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
            return CameraParams(
                CameraModel(data["model"]),
                tuple(data["dims"]),
                np.array(data["k"]),
                np.array(data["d"]),
            )


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
        atexit.register(self._cap.release)

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
        cv2.putText(img, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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


class Hyperion:
    _V4L_COMPONENT = "V4L"
    _V4L_OFF_DELAY_SEC = 5

    def __init__(self, api_url: str):
        self._api_url = api_url
        self._v4l_check()

    def _request(self, body):
        res = requests.post(
            self._api_url,
            json=body,
        )
        res.raise_for_status()
        return res.json()

    def get_serverinfo(self):
        return self._request({"command": "serverinfo"})["info"]

    def get_component(self, name: str):
        components = self.get_serverinfo()["components"]
        return next(c["enabled"] for c in components if c["name"] == name)

    def set_component(self, name: str, state: bool):
        self._request(
            {
                "command": "componentstate",
                "componentstate": {"component": name, "state": state},
            }
        )

    def _v4l_check(self):
        """disables V4L component for the duration of the program"""
        if not self.get_component(self._V4L_COMPONENT):
            return

        self.set_component(self._V4L_COMPONENT, False)
        atexit.register(self.set_component, self._V4L_COMPONENT, True)

        print(f"Waiting for shutdown of Hyperion {self._V4L_COMPONENT} component ...")
        sleep(self._V4L_OFF_DELAY_SEC)

    def get_config(self):
        return self._request({"command": "config", "subcommand": "getconfig"})["info"]

    def set_config(self, config):
        self._request(
            {
                "command": "config",
                "subcommand": "setconfig",
                "config": config,
            }
        )
