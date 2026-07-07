#!/usr/bin/env python3
"""Live UDP mosaic receiver for RK board preview frames.

The boards send packets created by comms.VideoSender:
  byte 0: message type (1 = video)
  byte 1: camera id
  byte 2-9: board id, zero padded
  remaining bytes: JPEG frame
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np

MSG_VIDEO = 0x01
MSG_VIDEO_CHUNK = 0x02
HEADER = struct.Struct("!BB8s")
CHUNK_HEADER = struct.Struct("!BB8sHHH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive board UDP preview and show a live mosaic.")
    parser.add_argument("--host", default="0.0.0.0", help="Local bind address.")
    parser.add_argument("--port", type=int, default=9999, help="UDP port.")
    parser.add_argument("--out-dir", default=r"E:\detect uav\record_2k_pure\board_video_receiver")
    parser.add_argument("--record", action="store_true", help="Record per-stream mp4 files.")
    parser.add_argument("--mosaic-record", action="store_true", help="Record the mosaic mp4.")
    parser.add_argument("--fps", type=float, default=8.0, help="Recording FPS.")
    parser.add_argument("--codec", default="MJPG", help="FourCC for recordings, e.g. MJPG or mp4v.")
    parser.add_argument("--ext", default="avi", help="Recording extension, e.g. avi or mp4.")
    parser.add_argument("--tile-w", type=int, default=320)
    parser.add_argument("--tile-h", type=int, default=180)
    parser.add_argument("--cols", type=int, default=5, help="Mosaic columns.")
    parser.add_argument("--max-streams", type=int, default=20)
    parser.add_argument("--stale-sec", type=float, default=2.0)
    parser.add_argument("--display-fps", type=float, default=15.0, help="Maximum mosaic refresh FPS.")
    parser.add_argument("--wait-ms", type=int, default=1, help="cv2.waitKey delay in milliseconds.")
    parser.add_argument("--rcvbuf-mb", type=float, default=4.0, help="UDP receive buffer size in MB. Smaller reduces latency.")
    parser.add_argument("--decode-budget-ms", type=float, default=30.0, help="Maximum packet decode time before refreshing.")
    parser.add_argument("--chunk-ttl-sec", type=float, default=0.5, help="Drop incomplete chunked frames older than this.")
    parser.add_argument(
        "--max-packets-per-loop",
        type=int,
        default=2000,
        help="Limit UDP packets decoded before refreshing the window.",
    )
    parser.add_argument(
        "--record-streams",
        default="",
        help="Comma separated streams to record, e.g. board2:2,board3:2. Empty means all.",
    )
    parser.add_argument(
        "--display-streams",
        default="",
        help="Comma separated streams to display, e.g. board2:2,board3:2. Empty means all.",
    )
    parser.add_argument("--no-window", action="store_true", help="Receive/record without showing a window.")
    return parser.parse_args()


def decode_board_id(raw: bytes) -> str:
    return raw.rstrip(b"\x00").decode("utf-8", "ignore") or "board"


def stream_sort_key(key: tuple[str, int]) -> tuple[int, int, str]:
    board, cam = key
    digits = "".join(ch for ch in board if ch.isdigit())
    board_num = int(digits) if digits else 999
    return board_num, cam, board


def parse_stream_filter(spec: str) -> set[tuple[str, int]] | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    allowed: set[tuple[str, int]] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            board, cam = token.split(":", 1)
        elif "_cam" in token:
            board, cam = token.split("_cam", 1)
        else:
            raise ValueError(f"bad stream spec: {token!r}")
        allowed.add((board.strip(), int(cam.strip())))
    return allowed


def stream_allowed(key: tuple[str, int], allowed: set[tuple[str, int]] | None) -> bool:
    return allowed is None or key in allowed


def seq_newer(a: int, b: int) -> bool:
    return a != b and ((a - b) & 0xFFFF) < 0x8000


def make_writer(path: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec[:4]), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path} codec={codec} size={size}")
    return writer


def draw_tile(
    tile: np.ndarray,
    key: tuple[str, int] | None,
    frame: np.ndarray | None,
    count: int,
    last_ts: float,
    stale_sec: float,
) -> np.ndarray:
    now = time.monotonic()
    if frame is None:
        tile[:] = (24, 24, 24)
        label = "waiting"
        color = (80, 80, 80)
    else:
        if frame.shape[1] == tile.shape[1] and frame.shape[0] == tile.shape[0]:
            tile[:] = frame
        else:
            resized = cv2.resize(frame, (tile.shape[1], tile.shape[0]), interpolation=cv2.INTER_AREA)
            tile[:] = resized
        age = now - last_ts
        label = f"{key[0]} cam{key[1]}  #{count}  {age:.1f}s"
        color = (0, 255, 0) if age <= stale_sec else (0, 160, 255)
        if age > stale_sec:
            overlay = tile.copy()
            overlay[:] = (20, 20, 20)
            cv2.addWeighted(overlay, 0.45, tile, 0.55, 0, tile)
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), color, 1)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return tile


def main() -> None:
    args = parse_args()
    record_filter = parse_stream_filter(args.record_streams)
    display_filter = parse_stream_filter(args.display_streams)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    requested_rcvbuf = max(64 * 1024, int(float(args.rcvbuf_mb) * 1024 * 1024))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested_rcvbuf)
    sock.bind((args.host, args.port))
    sock.setblocking(False)

    latest: dict[tuple[str, int], np.ndarray] = {}
    last_ts: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    writers: dict[tuple[str, int], cv2.VideoWriter] = {}
    chunks: dict[tuple[str, int, int], dict[str, object]] = {}
    latest_chunk_frame: dict[tuple[str, int], int] = {}
    mosaic_writer: cv2.VideoWriter | None = None
    stamp = time.strftime("%Y%m%d_%H%M%S")

    rows = max(1, math.ceil(args.max_streams / max(1, args.cols)))
    mosaic_size = (args.cols * args.tile_w, rows * args.tile_h)
    print(f"listening udp://{args.host}:{args.port}", flush=True)
    print(f"mosaic={mosaic_size[0]}x{mosaic_size[1]} cols={args.cols} max_streams={args.max_streams}", flush=True)
    print(
        f"low_latency rcvbuf={sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)//1024}KB "
        f"display_fps={args.display_fps} max_packets={args.max_packets_per_loop} "
        f"decode_budget={args.decode_budget_ms}ms wait={args.wait_ms}ms",
        flush=True,
    )
    print(f"record_streams={args.record_streams or 'all'} display_streams={args.display_streams or 'all'}", flush=True)
    print("keys: q/ESC quit, r toggle record, s save mosaic snapshot", flush=True)

    if not args.no_window:
        cv2.namedWindow("UAV board receiver", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("UAV board receiver", min(mosaic_size[0], 1800), min(mosaic_size[1], 1000))

    recording = bool(args.record)
    mosaic_recording = bool(args.mosaic_record)
    last_stats = time.monotonic()
    last_draw = 0.0
    display_period = 1.0 / max(1.0, float(args.display_fps))
    packets = 0

    try:
        while True:
            packets_this_loop = 0
            loop_start = time.monotonic()
            while packets_this_loop < max(1, int(args.max_packets_per_loop)):
                if (time.monotonic() - loop_start) * 1000.0 >= max(1.0, float(args.decode_budget_ms)):
                    break
                try:
                    packet, _addr = sock.recvfrom(65535)
                except BlockingIOError:
                    break
                if len(packet) <= HEADER.size:
                    continue
                packets_this_loop += 1
                msg_type = packet[0]
                if msg_type == MSG_VIDEO:
                    msg_type, cam_id, board_raw = HEADER.unpack(packet[: HEADER.size])
                    board_id = decode_board_id(board_raw)
                    jpeg_bytes = packet[HEADER.size :]
                elif msg_type == MSG_VIDEO_CHUNK:
                    if len(packet) <= CHUNK_HEADER.size:
                        continue
                    msg_type, cam_id, board_raw, frame_id, chunk_idx, chunk_count = CHUNK_HEADER.unpack(
                        packet[: CHUNK_HEADER.size]
                    )
                    board_id = decode_board_id(board_raw)
                    stream_key = (board_id, int(cam_id))
                    frame_id = int(frame_id)
                    known_frame_id = latest_chunk_frame.get(stream_key)
                    if known_frame_id is None or seq_newer(frame_id, known_frame_id):
                        latest_chunk_frame[stream_key] = frame_id
                        for old_key in [k for k in chunks if k[0] == board_id and k[1] == int(cam_id)]:
                            chunks.pop(old_key, None)
                    elif frame_id != known_frame_id:
                        continue
                    chunk_key = (board_id, int(cam_id), frame_id)
                    state = chunks.get(chunk_key)
                    now = time.monotonic()
                    if state is None:
                        state = {"count": int(chunk_count), "parts": {}, "ts": now}
                        chunks[chunk_key] = state
                    state["ts"] = now
                    parts = state["parts"]
                    parts[int(chunk_idx)] = packet[CHUNK_HEADER.size :]
                    if len(parts) < int(state["count"]):
                        if len(chunks) > 64:
                            old_keys = [
                                k for k, v in chunks.items()
                                if now - float(v.get("ts", now)) > float(args.chunk_ttl_sec)
                            ]
                            for old_key in old_keys:
                                chunks.pop(old_key, None)
                        continue
                    jpeg_bytes = b"".join(parts[i] for i in range(int(state["count"])))
                    chunks.pop(chunk_key, None)
                else:
                    continue
                img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                key = (board_id, int(cam_id))
                if stream_allowed(key, display_filter):
                    latest[key] = img
                    last_ts[key] = time.monotonic()
                    counts[key] = counts.get(key, 0) + 1
                packets += 1

                if recording and stream_allowed(key, record_filter):
                    writer = writers.get(key)
                    if writer is None:
                        stream_dir = out_dir / key[0] / f"cam{key[1]}"
                        stream_dir.mkdir(parents=True, exist_ok=True)
                        h, w = img.shape[:2]
                        path = stream_dir / f"{key[0]}_cam{key[1]}_{stamp}.{args.ext.lstrip('.')}"
                        writer = make_writer(path, args.fps, (w, h), args.codec)
                        writers[key] = writer
                        print(f"recording {path}", flush=True)
                    writer.write(img)

            keys = sorted(latest.keys(), key=stream_sort_key)[: args.max_streams]
            now = time.monotonic()
            if not args.no_window and not mosaic_recording and now - last_draw < display_period:
                time.sleep(min(0.005, display_period - (now - last_draw)))
                continue
            last_draw = time.monotonic()
            mosaic = np.zeros((mosaic_size[1], mosaic_size[0], 3), dtype=np.uint8)
            for idx in range(args.max_streams):
                row = idx // args.cols
                col = idx % args.cols
                tile = mosaic[
                    row * args.tile_h : (row + 1) * args.tile_h,
                    col * args.tile_w : (col + 1) * args.tile_w,
                ]
                key = keys[idx] if idx < len(keys) else None
                if key is None:
                    draw_tile(tile, None, None, 0, 0.0, args.stale_sec)
                else:
                    draw_tile(tile, key, latest.get(key), counts.get(key, 0), last_ts.get(key, 0.0), args.stale_sec)

            if mosaic_recording:
                if mosaic_writer is None:
                    path = out_dir / f"mosaic_{stamp}.{args.ext.lstrip('.')}"
                    mosaic_writer = make_writer(path, args.fps, mosaic_size, args.codec)
                    print(f"recording {path}", flush=True)
                mosaic_writer.write(mosaic)

            now = time.monotonic()
            if now - last_stats >= 3.0:
                summary = ", ".join(f"{b}_cam{c}:{counts[(b, c)]}" for b, c in keys[:10])
                more = "" if len(keys) <= 10 else f" ... +{len(keys)-10}"
                print(f"streams={len(keys)} packets={packets} {summary}{more}", flush=True)
                last_stats = now

            if args.no_window:
                time.sleep(0.02)
                continue
            cv2.imshow("UAV board receiver", mosaic)
            key_code = cv2.waitKey(max(1, int(args.wait_ms))) & 0xFF
            if key_code in (27, ord("q")):
                break
            if key_code == ord("r"):
                recording = not recording
                print(f"recording={'on' if recording else 'off'}", flush=True)
            if key_code == ord("s"):
                path = out_dir / f"mosaic_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), mosaic)
                print(f"saved {path}", flush=True)
    finally:
        for writer in writers.values():
            writer.release()
        if mosaic_writer is not None:
            mosaic_writer.release()
        sock.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        print("summary:", {f"{b}_cam{c}": counts[(b, c)] for b, c in sorted(counts, key=stream_sort_key)}, flush=True)


if __name__ == "__main__":
    main()
