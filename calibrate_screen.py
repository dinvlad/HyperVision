#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import sys

import cv2
import numpy as np
from screeninfo import Monitor, get_monitors

from util import (
    CameraModel,
    CameraParams,
    Hyperion,
    VideoCapture,
    arg_non_neg_int,
    arg_pos_int,
    arg_scale_float,
    imshow,
    show_preview,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate screen LED capture areas for Hyperion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-sm",
        "--screen-monitor",
        type=arg_non_neg_int,
        default=0,
        help="Which monitor number to use, for multi-monitor setups",
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
        "--frame-preview-delay-ms",
        type=arg_pos_int,
        default=1,
        help="Minimum delay between preview camera captures, in milliseconds",
    )
    parser.add_argument(
        "-fs",
        "--frame-blank-delay-ms",
        type=arg_pos_int,
        default=1000,
        help="Delay before capturing the blank frame, in milliseconds",
    )
    parser.add_argument(
        "-ci",
        "--cam-index",
        type=arg_non_neg_int,
        default=0,
        help="Index of the camera capture device",
    )
    parser.add_argument(
        "-cf",
        "--cam-fps",
        type=arg_pos_int,
        default=5,
        help="Frames-per-second for camera capture, in pixels (used only during calibration)",
    )
    parser.add_argument(
        "-cp",
        "--cam-params-file",
        type=str,
        default="params.json",
        help="Path to the JSON file that stores camera parameters",
    )
    parser.add_argument(
        "-ca",
        "--cam-alpha",
        type=arg_scale_float,
        default=0.0,
        help="""
            Alpha/balance scaling parameter between 0 (when all the pixels in the undistorted camera image are valid)
            and 1 (when all the source image pixels are retained in the undistorted image)
        """,
    )
    parser.add_argument(
        "-lt",
        "--leds-top",
        type=arg_pos_int,
        default=96,
        help="Number of LEDs at the top of the screen",
    )
    parser.add_argument(
        "-lr",
        "--leds-right",
        type=arg_pos_int,
        default=54,
        help="Number of LEDs at the right of the screen",
    )
    parser.add_argument(
        "-lb",
        "--leds-bottom",
        type=arg_pos_int,
        default=96,
        help="Number of LEDs at the bottom of the screen",
    )
    parser.add_argument(
        "-ll",
        "--leds-left",
        type=arg_pos_int,
        default=54,
        help="Number of LEDs at the left of the screen",
    )
    parser.add_argument(
        "-lh",
        "--led-depth-horiz-pct",
        type=int,
        choices=range(1, 101),
        metavar="[1-100]",
        default=8,
        help="Percent depth of LED capture area at the top and bottom of the screen",
    )
    parser.add_argument(
        "-lv",
        "--led-depth-vert-pct",
        type=int,
        choices=range(1, 101),
        metavar="[1-100]",
        default=5,
        help="Percent depth of LED capture area at the left and right of the screen",
    )
    parser.add_argument(
        "-ha",
        "--hyperion-api",
        type=str,
        default="http://localhost:8090/json-rpc",
        help="URL of Hyperion HTTP/S JSON API",
    )
    return parser.parse_args()


def get_blank_frame(mon: Monitor, cap: VideoCapture, wait_ms: int):
    blank = np.full((mon.height, mon.width), 255, dtype=np.uint8)

    window = "Calibration"
    cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window, blank)
    cv2.waitKey(wait_ms)

    blank = cap.read()

    cv2.destroyWindow(window)

    return blank


@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int


def get_screen_transforms(blank: np.array, cam: CameraParams, cam_alpha: float):
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea)
    cnt = contours[-1]
    crop = Crop(*cv2.boundingRect(cnt))

    epsilon = 0
    epsilon_inc = 0.001
    approx = cnt
    cnt_len = cv2.arcLength(cnt, True)

    while len(approx) > 4:
        epsilon += epsilon_inc
        approx = cv2.approxPolyDP(cnt, epsilon * cnt_len, True)

    hull = cv2.convexHull(approx, clockwise=False)
    i = np.argmin([np.linalg.norm(p[0]) for p in hull])
    corners = np.roll(hull, -i, axis=0)
    corners = np.float32(corners)

    if cam.model == CameraModel.PINHOLE:
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            cam.k,
            cam.d,
            cam.dims,
            cam_alpha,
        )
        undist_points = cv2.undistortPoints
    elif cam.model == CameraModel.FISHEYE:
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            cam.k,
            cam.d,
            cam.dims,
            np.eye(3),
            balance=cam_alpha,
        )
        undist_points = cv2.fisheye.undistortPoints

    new_k_inv = np.linalg.inv(new_k)
    undist_corners = undist_points(corners, cam.k, cam.d, None, new_k)

    scr_corners = np.float32([[[0, 0]], [[1, 0]], [[1, 1]], [[0, 1]]])
    p_t = cv2.getPerspectiveTransform(scr_corners, undist_corners)

    return crop, new_k_inv, p_t


def distort_points(
    cam: CameraParams,
    points: np.array,
    p_t: np.array,
    new_k_inv: np.array,
):
    perspective_points = cv2.perspectiveTransform(points, p_t)
    perspective_points_3d = (
        cv2.convertPointsToHomogeneous(perspective_points).transpose().squeeze()
    )
    perspective_points_3d = np.matmul(new_k_inv, perspective_points_3d)
    perspective_points = cv2.convertPointsFromHomogeneous(
        perspective_points_3d.transpose()[None, :, :]
    )
    if cam.model == CameraModel.PINHOLE:
        distorted_points, _ = cv2.projectPoints(
            perspective_points, np.zeros(3), np.zeros(3), cam.k, cam.d
        )
        return distorted_points.squeeze()
    elif cam.model == CameraModel.FISHEYE:
        return cv2.fisheye.distortPoints(perspective_points, cam.k, cam.d).squeeze()


def calculate_leds(
    cam: CameraParams,
    new_k_inv: np.array,
    p_t: np.array,
    leds_top: int,
    leds_right: int,
    leds_bottom: int,
    leds_left: int,
    horiz_depth: int,
    vert_depth: int,
):
    leds = []

    def distort(points: list):
        return distort_points(cam, np.float32(points), p_t, new_k_inv)

    dp = distort([[[i / leds_top, 0]] for i in range(0, leds_top + 1)])
    for i in range(0, len(dp) - 1):
        y = max(dp[i][1], dp[i + 1][1])
        leds.append((dp[i][0], dp[i + 1][0], y, y + horiz_depth))

    dp = distort([[[1, i / leds_right]] for i in range(0, leds_right + 1)])
    for i in range(0, len(dp) - 1):
        x = min(dp[i][0], dp[i + 1][0]) - vert_depth
        leds.append((x, x + vert_depth, dp[i][1], dp[i + 1][1]))

    dp = distort([[[1 - i / leds_bottom, 1]] for i in range(0, leds_bottom + 1)])
    for i in range(0, len(dp) - 1):
        y = min(dp[i][1], dp[i + 1][1]) - horiz_depth
        leds.append((dp[i + 1][0], dp[i][0], y, y + horiz_depth))

    dp = distort([[[0, 1 - i / leds_left]] for i in range(0, leds_left + 1)])
    for i in range(0, len(dp) - 1):
        x = max(dp[i][0], dp[i + 1][0])
        leds.append((x, x + vert_depth, dp[i + 1][1], dp[i][1]))

    return leds


def show_leds(
    mon: Monitor,
    cap: VideoCapture,
    leds: np.array,
    crop: Crop,
    frame_delay_ms: int,
):
    img = None
    while img is None:
        img = cap.read()
        cv2.waitKey(frame_delay_ms)

    y0 = round((mon.height - img.shape[0]) / 2)
    x0 = round((mon.width - img.shape[1]) / 2)

    for p in leds:
        cv2.rectangle(img, np.int32((p[0], p[2])), np.int32((p[1], p[3])), (255, 0, 0))

    img = img[crop.y : crop.y + crop.h, crop.x : crop.x + crop.w, :]

    imshow(
        "LED Preview",
        img,
        (y0, x0),
        img.shape[:2],
        "Press any key to update Hyperion, ESC to cancel",
    )

    if cv2.waitKey(0) == 0x1B:
        sys.exit(1)


def update_hyperion(
    hyperion: Hyperion,
    cam: CameraParams,
    leds: np.array,
    crop: Crop,
):
    config = hyperion.get_config()

    if len(leds) > config["device"]["hardwareLedCount"]:
        raise ValueError(
            "The total number of LEDs must be less than 'Hardware LED count' in Hyperion settings!"
        )

    config["leds"] = [
        {
            "hmin": np.clip((led[0] - crop.x) / crop.w, 0, 1),
            "hmax": np.clip((led[1] - crop.x) / crop.w, 0, 1),
            "vmin": np.clip((led[2] - crop.y) / crop.h, 0, 1),
            "vmax": np.clip((led[3] - crop.y) / crop.h, 0, 1),
        }
        for led in leds
    ]

    grabber = config["grabberV4L2"]
    grabber["width"] = cam.dims[0]
    grabber["height"] = cam.dims[1]
    grabber["cropTop"] = crop.y
    grabber["cropRight"] = cam.dims[0] - crop.w - crop.x
    grabber["cropBottom"] = cam.dims[1] - crop.h - crop.y
    grabber["cropLeft"] = crop.x

    hyperion.set_config(config)


def main():
    args = parse_args()

    cam = CameraParams.load(args.cam_params_file)
    mon = get_monitors()[args.screen_monitor]

    if args.hyperion_api:
        hyperion = Hyperion(args.hyperion_api)

    cap = VideoCapture(args.cam_index, cam.dims[::-1], args.cam_fps)
    show_preview(cap, mon, args.screen_preview_scale, args.frame_preview_delay_ms)

    blank = get_blank_frame(mon, cap, args.frame_blank_delay_ms)
    crop, new_k_inv, p_t = get_screen_transforms(blank, cam, args.cam_alpha)

    horiz_led_depth = crop.h * args.led_depth_horiz_pct / 100
    vert_led_depth = crop.w * args.led_depth_vert_pct / 100

    leds = calculate_leds(
        cam,
        new_k_inv,
        p_t,
        args.leds_top,
        args.leds_right,
        args.leds_bottom,
        args.leds_left,
        horiz_led_depth,
        vert_led_depth,
    )

    show_leds(mon, cap, leds, crop, args.frame_preview_delay_ms)
    update_hyperion(hyperion, cam, leds, crop)


if __name__ == "__main__":
    main()
