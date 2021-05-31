#!/usr/bin/env python3

import argparse
from enum import Enum
import json
from secrets import randbelow
from time import time
from typing import Tuple

import cv2
import numpy as np
from screeninfo import Monitor, get_monitors

from util import (
    VideoCapture,
    arg_pos_float,
    arg_scale_float,
    arg_non_neg_int,
    arg_pos_int,
    imshow,
    show_preview,
)


class CameraModel(Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate a fisheye camera.")
    parser.add_argument(
        "-br",
        "--board-rows",
        type=arg_pos_int,
        default=9,
        help="Number of calibration board rows",
    )
    parser.add_argument(
        "-bc",
        "--board-cols",
        type=arg_pos_int,
        default=6,
        help="Number of calibration board columns",
    )
    parser.add_argument(
        "-bf",
        "--board-frames",
        type=arg_pos_int,
        default=20,
        help="How many frames to use for the calibration",
    )
    parser.add_argument(
        "-sm",
        "--screen-monitor",
        type=arg_non_neg_int,
        default=0,
        help="Which monitor number to use, for multi-monitor setups",
    )
    parser.add_argument(
        "-sb",
        "--screen-board-scale",
        type=arg_scale_float,
        default=0.5,
        help="Scale of the calibration board relative to screen size",
    )
    parser.add_argument(
        "-sp",
        "--screen-preview-scale",
        type=arg_scale_float,
        default=0.5,
        help="Scale of the camera preview window",
    )
    parser.add_argument(
        "-fd",
        "--frame-delay-ms",
        type=arg_pos_int,
        default=1,
        help="Minimum delay between camera captures, in milliseconds",
    )
    parser.add_argument(
        "-fs",
        "--frame-success-delay-ms",
        type=arg_pos_int,
        default=1000,
        help="Minimum delay between successful camera captures, in milliseconds",
    )
    parser.add_argument(
        "-ci",
        "--cam-index",
        type=arg_non_neg_int,
        default=0,
        help="Index of the camera capture device",
    )
    parser.add_argument(
        "-cx",
        "--cam-res-x",
        type=arg_pos_int,
        default=1280,
        help="Horizontal resolution for camera capture, in pixels",
    )
    parser.add_argument(
        "-cy",
        "--cam-res-y",
        type=arg_pos_int,
        default=720,
        help="Vertical resolution for camera capture, in pixels",
    )
    parser.add_argument(
        "-cf",
        "--cam-fps",
        type=arg_pos_int,
        default=5,
        help="Frames-per-second for camera capture, in pixels (used only during calibration)",
    )
    parser.add_argument(
        "-cm",
        "--cam-model",
        type=CameraModel,
        choices=list(CameraModel),
        default=CameraModel.PINHOLE,
        help="Mathematical model of the camera to use",
    )
    parser.add_argument(
        "-ic",
        "--iter-calib",
        type=arg_pos_int,
        default=30,
        help="Maximum iterations for camera calibration",
    )
    parser.add_argument(
        "-is",
        "--iter-subpix",
        type=arg_pos_int,
        default=30,
        help="Maximum iterations for corner subpixel calculation",
    )
    parser.add_argument(
        "-ec",
        "--epsilon-calib",
        type=arg_pos_float,
        default=1e-6,
        help="Epsilon accuracy for camera calibration",
    )
    parser.add_argument(
        "-es",
        "--epsilon-subpix",
        type=arg_pos_float,
        default=1e-3,
        help="Epsilon accuracy for corner subpixel calculation",
    )
    parser.add_argument(
        "-o",
        "--output-params-file",
        type=str,
        default="params.json",
        help="Path to the JSON file for storing the output parameters",
    )
    return parser.parse_args()


def get_board_img(mon: Monitor, board_dims: Tuple[int, int], screen_board_scale: float):
    board_square = min(
        round(mon.height * screen_board_scale / (board_dims[0] + 3)),
        round(mon.width * screen_board_scale / (board_dims[1] + 3)),
    )
    img_height = board_square * (board_dims[0] + 3)
    img_width = board_square * (board_dims[1] + 3)

    black = 0
    white = 255

    board_img = np.full((img_height, img_width), white, np.uint8)

    black_row = False
    black_col = False

    for row in range(1, board_dims[0] + 2):
        black_row = not black_row
        black_col = black_row

        y = row * board_square

        for col in range(1, board_dims[1] + 2):
            x = col * board_square

            board_img[y : y + board_square, x : x + board_square] = (
                black if black_col else white
            )
            black_col = not black_col

    return board_img


def show_board(mon: Monitor, board_img: np.array, delay_ms: int):
    y0 = randbelow(mon.height - board_img.shape[0])
    x0 = randbelow(mon.width - board_img.shape[1])
    imshow("Calibration", board_img, (y0, x0), board_img.shape)
    cv2.waitKey(delay_ms)


def calibrate(
    cap: VideoCapture,
    mon: Monitor,
    cam_model: CameraModel,
    board_dims: Tuple[int, int],
    board_img: np.array,
    board_frames: int,
    frame_delay_ms: int,
    frame_success_delay_ms: int,
    iter_calib: int,
    iter_subpix: int,
    epsilon_calib: float,
    epsilon_subpix: float,
):
    board_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if cam_model == CameraModel.PINHOLE:
        calib_func = cv2.calibrateCamera
        calib_flags = 0
    elif cam_model == CameraModel.FISHEYE:
        calib_func = cv2.fisheye.calibrate
        calib_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        )
    term_criteria = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER
    calib_criteria = (term_criteria, iter_calib, epsilon_calib)
    subpix_criteria = (term_criteria, iter_subpix, epsilon_subpix)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    objp = np.zeros((1, board_dims[0] * board_dims[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : board_dims[0], 0 : board_dims[1]].T.reshape(-1, 2)

    found = True

    while len(imgpoints) < board_frames:
        if found or (time() - start) * 1000 > frame_success_delay_ms:
            show_board(mon, board_img, frame_success_delay_ms)
            start = time()

        img = cap.read()
        if img is None:
            cv2.waitKey(frame_delay_ms)
            continue

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            img_gray,
            board_dims,
            board_flags,
        )

        if found:
            cv2.cornerSubPix(img_gray, corners, (3, 3), (-1, -1), subpix_criteria)
            cv2.drawChessboardCorners(img, board_dims, corners, found)

            objpoints.append(objp)
            imgpoints.append(corners)

            print("Detected frame: ", len(imgpoints), "out of", board_frames)

    dims = img_gray.shape[::-1]
    rms, k, d, _, _ = calib_func(
        objpoints,
        imgpoints,
        dims,
        None,
        None,
        flags=calib_flags,
        criteria=calib_criteria,
    )

    print(f"Used {len(objpoints)} valid images for calibration. RMS error: {rms}")

    return {
        "model": str(cam_model),
        "dims": dims,
        "k": k.tolist(),
        "d": d.tolist(),
    }


def main():
    args = parse_args()

    mon = get_monitors()[args.screen_monitor]

    cap = VideoCapture(args.cam_index, (args.cam_res_y, args.cam_res_x), args.cam_fps)
    show_preview(cap, mon, args.screen_preview_scale, args.frame_delay_ms)

    board_dims = (args.board_rows, args.board_cols)
    board_img = get_board_img(mon, board_dims, args.screen_board_scale)

    params = calibrate(
        cap,
        mon,
        args.cam_model,
        board_dims,
        board_img,
        args.board_frames,
        args.frame_delay_ms,
        args.frame_success_delay_ms,
        args.iter_calib,
        args.iter_subpix,
        args.epsilon_calib,
        args.epsilon_subpix,
    )

    with open(args.output_params_file, "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
