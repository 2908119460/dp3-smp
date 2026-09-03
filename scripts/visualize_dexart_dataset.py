#!/usr/bin/env python3
"""Render one multimodal quality-review video per DexArt episode."""

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import zarr


TASKS = ("faucet", "toilet", "bucket", "laptop")
FRAME_SIZE = (1280, 720)
FPS = 10
BG = (24, 21, 18)
PANEL_BG = (36, 32, 28)
TEXT = (235, 235, 235)
MUTED = (165, 165, 165)
ACCENT = (80, 205, 255)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "3D-Diffusion-Policy",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--variant",
        default="expert",
        help="dataset suffix, e.g. 'expert' or 'new'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crf", type=int, default=23)
    return parser.parse_args()


def put_text(image, text, origin, scale=0.55, color=TEXT, thickness=1):
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def panel(image, rect, title):
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL_BG, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (72, 67, 62), 1)
    put_text(image, title, (x + 10, y + 23), 0.52, ACCENT, 1)


def fit_image(source, width, height, interpolation=cv2.INTER_NEAREST):
    src_height, src_width = source.shape[:2]
    scale = min(width / src_width, height / src_height)
    new_size = (max(1, int(src_width * scale)), max(1, int(src_height * scale)))
    resized = cv2.resize(source, new_size, interpolation=interpolation)
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    x = (width - new_size[0]) // 2
    y = (height - new_size[1]) // 2
    canvas[y:y + new_size[1], x:x + new_size[0]] = resized
    return canvas


def projection(points):
    angle = math.radians(42.0)
    ca, sa = math.cos(angle), math.sin(angle)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rotated_x = ca * x - sa * y
    rotated_y = sa * x + ca * y
    u = rotated_x
    v = z + 0.28 * rotated_y
    depth = rotated_y
    return np.stack((u, v, depth), axis=-1)


def compute_task_scale(group):
    point_cloud = group["data"]["point_cloud"]
    robot = group["data"]["imagin_robot"]
    depth = group["data"]["depth"]
    indices = np.linspace(0, len(point_cloud) - 1, min(96, len(point_cloud)), dtype=int)
    pc_sample = np.asarray(point_cloud.oindex[indices])[..., :3].reshape(-1, 3)
    robot_sample = np.asarray(robot.oindex[indices])[..., :3].reshape(-1, 3)
    projected = projection(np.concatenate((pc_sample, robot_sample), axis=0))
    low = np.nanpercentile(projected[:, :2], 0.5, axis=0)
    high = np.nanpercentile(projected[:, :2], 99.5, axis=0)
    padding = np.maximum((high - low) * 0.08, 1e-3)

    depth_sample = np.asarray(depth.oindex[indices]).reshape(-1)
    valid_depth = depth_sample[np.isfinite(depth_sample) & (depth_sample > 0)]
    if valid_depth.size:
        depth_range = np.nanpercentile(valid_depth, (1.0, 99.0))
    else:
        depth_range = np.array((0.0, 1.0))
    return low - padding, high + padding, depth_range


def render_depth(depth, depth_range, width, height):
    low, high = depth_range
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if high > low:
        values = 1.0 - np.clip((depth - low) / (high - low), 0.0, 1.0)
        normalized[valid] = np.asarray(values[valid] * 255, dtype=np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return fit_image(colored, width, height, cv2.INTER_NEAREST)


def render_point_cloud(point_cloud, robot, bounds, width, height):
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    low, high = bounds
    projected = projection(point_cloud[:, :3])
    finite = np.all(np.isfinite(projected), axis=1)
    projected = projected[finite]
    colors = point_cloud[finite, 3:6] if point_cloud.shape[1] >= 6 else None
    if colors is None or colors.size == 0:
        colors = np.tile(np.array([[210, 210, 210]], dtype=np.float32), (len(projected), 1))
    if np.nanmax(colors, initial=0.0) <= 1.5:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)[:, ::-1]

    span = np.maximum(high - low, 1e-6)
    px = ((projected[:, 0] - low[0]) / span[0] * (width - 24) + 12).astype(int)
    py = ((1.0 - (projected[:, 1] - low[1]) / span[1]) * (height - 24) + 12).astype(int)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    order = np.argsort(projected[:, 2])
    for index in order:
        if inside[index]:
            cv2.circle(canvas, (px[index], py[index]), 1, colors[index].tolist(), -1)

    robot_projected = projection(robot[:, :3])
    robot_finite = np.all(np.isfinite(robot_projected), axis=1)
    robot_projected = robot_projected[robot_finite]
    rx = ((robot_projected[:, 0] - low[0]) / span[0] * (width - 24) + 12).astype(int)
    ry = ((1.0 - (robot_projected[:, 1] - low[1]) / span[1]) * (height - 24) + 12).astype(int)
    for x, y in zip(rx, ry):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 2, (255, 80, 220), -1)
    return canvas


def action_heatmap(actions, frame_index, width, height):
    action_values = np.clip(actions.T, -1.0, 1.0)
    red = np.clip(action_values, 0.0, 1.0) * 215
    blue = np.clip(-action_values, 0.0, 1.0) * 215
    neutral = 48 + (1.0 - np.abs(action_values)) * 42
    heatmap = np.stack(
        (neutral + blue, neutral, neutral + red), axis=-1
    ).clip(0, 255).astype(np.uint8)
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    if len(actions) > 1:
        cursor = int(frame_index / (len(actions) - 1) * (width - 1))
    else:
        cursor = 0
    cv2.line(resized, (cursor, 0), (cursor, height - 1), (255, 255, 255), 2)
    return resized


def build_frame(task, episode_index, frame_index, episode, task_scale):
    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), BG, dtype=np.uint8)
    length = len(episode["img"])
    put_text(frame, f"DexArt dataset quality review | {task}", (20, 35), 0.78, TEXT, 2)
    put_text(
        frame,
        f"Episode {episode_index:03d} | Frame {frame_index + 1:03d}/{length:03d} | 10 FPS",
        (760, 35),
        0.58,
        MUTED,
        1,
    )

    rgb_rect = (20, 60, 390, 410)
    depth_rect = (430, 60, 390, 410)
    pc_rect = (840, 60, 420, 410)
    action_rect = (20, 490, 800, 195)
    info_rect = (840, 490, 420, 195)
    for rect, title in (
        (rgb_rect, "RGB observation"),
        (depth_rect, "Depth observation"),
        (pc_rect, "Point cloud + robot points"),
        (action_rect, "Action history (22 dimensions, blue=-1, red=+1)"),
        (info_rect, "Frame diagnostics"),
    ):
        panel(frame, rect, title)

    rgb = cv2.cvtColor(episode["img"][frame_index], cv2.COLOR_RGB2BGR)
    frame[95:455, 35:395] = fit_image(rgb, 360, 360, cv2.INTER_NEAREST)
    frame[95:455, 445:805] = render_depth(
        episode["depth"][frame_index], task_scale[2], 360, 360
    )
    frame[95:455, 855:1245] = render_point_cloud(
        episode["point_cloud"][frame_index],
        episode["imagin_robot"][frame_index],
        task_scale[:2],
        390,
        360,
    )
    frame[525:665, 35:805] = action_heatmap(
        episode["action"], frame_index, 770, 140
    )

    action = episode["action"][frame_index]
    point_cloud = episode["point_cloud"][frame_index, :, :3]
    finite_points = np.all(np.isfinite(point_cloud), axis=1)
    zero_points = np.all(np.isclose(point_cloud, 0.0), axis=1)
    if frame_index:
        action_delta = np.linalg.norm(action - episode["action"][frame_index - 1])
        state_delta = np.linalg.norm(
            episode["state"][frame_index] - episode["state"][frame_index - 1]
        )
    else:
        action_delta = 0.0
        state_delta = 0.0
    diagnostics = (
        f"Action L2:       {np.linalg.norm(action):7.3f}",
        f"Action max abs:  {np.max(np.abs(action)):7.3f}",
        f"Action delta:    {action_delta:7.3f}",
        f"State delta:     {state_delta:7.3f}",
        f"Finite points:   {int(finite_points.sum()):4d}/1024",
        f"Zero points:     {int(zero_points.sum()):4d}/1024",
    )
    for line_index, line in enumerate(diagnostics):
        put_text(frame, line, (860, 528 + line_index * 24), 0.50, TEXT, 1)

    progress = (frame_index + 1) / length
    cv2.rectangle(frame, (20, 700), (1260, 707), (58, 54, 50), -1)
    cv2.rectangle(frame, (20, 700), (20 + int(1240 * progress), 707), ACCENT, -1)
    return frame


def open_encoder(output_path, crf):
    command = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def render_episode(task, episode_index, group, start, end, task_scale, output_path, crf):
    data = group["data"]
    episode = {
        key: np.asarray(data[key][start:end])
        for key in ("img", "depth", "point_cloud", "imagin_robot", "action", "state")
    }
    encoder = open_encoder(output_path, crf)
    try:
        for frame_index in range(end - start):
            frame = build_frame(task, episode_index, frame_index, episode, task_scale)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait()
    except Exception:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.kill()
        encoder.wait()
        output_path.unlink(missing_ok=True)
        raise
    if return_code != 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output_path} with code {return_code}")


def main():
    args = parse_args()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    data_root = args.project_root / "data"
    output_root = args.project_root / "data_visualize"
    output_root.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        source = data_root / f"dexart_{task}_{args.variant}.zarr"
        output_suffix = "" if args.variant == "expert" else f"_{args.variant}"
        output_dir = output_root / f"dexart_{task}{output_suffix}"
        output_dir.mkdir(parents=True, exist_ok=True)
        group = zarr.open(str(source), mode="r")
        episode_ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
        task_scale = compute_task_scale(group)
        total = len(episode_ends) if args.limit is None else min(args.limit, len(episode_ends))
        print(f"[{task}] rendering {total} episodes", flush=True)
        for episode_index in range(total):
            output_path = output_dir / f"episode_{episode_index:03d}.mp4"
            if output_path.exists() and not args.overwrite:
                print(f"  skip {output_path.name}", flush=True)
                continue
            start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
            end = int(episode_ends[episode_index])
            print(
                f"  {episode_index + 1:03d}/{total:03d} {output_path.name} ({end - start} frames)",
                flush=True,
            )
            render_episode(
                task, episode_index, group, start, end, task_scale, output_path, args.crf
            )


if __name__ == "__main__":
    main()
