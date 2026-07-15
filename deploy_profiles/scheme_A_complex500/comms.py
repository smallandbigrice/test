import json
import socket
import struct

import cv2


MSG_VIDEO = 0x01
MSG_VIDEO_CHUNK = 0x02
MSG_INIT = 0x03

VIDEO_HEADER = struct.Struct("!BB8s")
VIDEO_CHUNK_HEADER = struct.Struct("!BB8sHHH")


class VideoSender:
    def __init__(self, ip, port, width=640, height=360, quality=50):
        self.ip = ip
        self.port = port
        self.width = int(width)
        self.height = int(height)
        self.quality = int(quality)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        self.max_udp_size = 60000
        self.frame_ids = {}

    def send(self, board_id, cam_id, frame):
        try:
            if frame is None:
                return
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                return
            jpeg_data = encoded.tobytes()

            b_id_bytes = str(board_id).encode("utf-8")[:8].ljust(8, b"\x00")
            full_packet = VIDEO_HEADER.pack(MSG_VIDEO, int(cam_id), b_id_bytes) + jpeg_data
            if len(full_packet) < self.max_udp_size:
                self.sock.sendto(full_packet, (self.ip, self.port))
                return

            stream_key = (str(board_id), int(cam_id))
            frame_id = (self.frame_ids.get(stream_key, 0) + 1) & 0xFFFF
            self.frame_ids[stream_key] = frame_id
            payload_size = self.max_udp_size - VIDEO_CHUNK_HEADER.size
            chunk_count = (len(jpeg_data) + payload_size - 1) // payload_size
            if chunk_count > 65535:
                return

            for chunk_idx in range(chunk_count):
                start = chunk_idx * payload_size
                end = start + payload_size
                header = VIDEO_CHUNK_HEADER.pack(
                    MSG_VIDEO_CHUNK,
                    int(cam_id),
                    b_id_bytes,
                    int(frame_id),
                    int(chunk_idx),
                    int(chunk_count),
                )
                self.sock.sendto(header + jpeg_data[start:end], (self.ip, self.port))
        except Exception:
            pass


class DataSender:
    def __init__(self, targets, board_id):
        self.targets = targets
        self.board_id = board_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_packet(self, msg_type, cam_id, content=None, target=None):
        if target not in self.targets:
            return

        ip, port = self.targets[target]
        payload = {
            "board": self.board_id,
            "cam": cam_id,
            "type": msg_type,
        }

        if msg_type == "data":
            payload["objs"] = content
        elif msg_type in ["status", "error"]:
            payload["msg"] = content

        try:
            msg = json.dumps(payload).encode("utf-8")
            self.sock.sendto(msg, (ip, port))
        except Exception:
            pass
