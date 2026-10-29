"""
Binary MessagePack landmark streaming WebSocket handler.
Replaces JPEG base64 with binary Float32Array (543-d landmarks) + MessagePack.
"""

from __future__ import annotations

import json
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

LANDMARK_DIM = 543
MAX_QUEUE_SIZE = 32
LATENCY_HORIZON = 100  # ms; drop frames behind this threshold


@dataclass
class StageMetrics:
    name: str
    latencies: deque = field(default_factory=lambda: deque(maxlen=300))

    def observe(self, ms: float) -> None:
        self.latencies.append(ms)

    def summary(self) -> dict[str, float]:
        if not self.latencies:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0}
        arr = np.array(self.latencies, dtype=np.float64)
        return {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "count": int(arr.size),
        }


@dataclass
class FrameRecord:
    client_ts: float
    server_recv_ts: float
    landmarks: np.ndarray
    seq: int


@dataclass
class LatencyMetrics:
    capture: StageMetrics = field(default_factory=lambda: StageMetrics("capture"))
    extract: StageMetrics = field(default_factory=lambda: StageMetrics("extract"))
    infer: StageMetrics = field(default_factory=lambda: StageMetrics("infer"))
    decode: StageMetrics = field(default_factory=lambda: StageMetrics("decode"))
    render: StageMetrics = field(default_factory=lambda: StageMetrics("render"))
    tts: StageMetrics = field(default_factory=lambda: StageMetrics("tts"))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "capture": self.capture.summary(),
            "extract": self.extract.summary(),
            "infer": self.infer.summary(),
            "decode": self.decode.summary(),
            "render": self.render.summary(),
            "tts": self.tts.summary(),
        }


class RecognitionConnection:
    """Manages a single WebSocket connection with binary landmark streaming."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.frame_queue: deque[FrameRecord] = deque(maxlen=MAX_QUEUE_SIZE)
        self.metrics = LatencyMetrics()
        self._seq = 0

    async def handle(self) -> None:
        await self.websocket.accept()
        try:
            while True:
                data = await self.websocket.receive_bytes()
                await self._process_frame(data)
        except WebSocketDisconnect:
            pass

    async def _process_frame(self, data: bytes) -> None:
        recv_ts = time.perf_counter()
        parsed = self._parse_binary_frame(data, recv_ts)
        if parsed is None:
            return
        self._enqueue_frame(parsed)
        await self._run_inference(parsed)

    def _parse_binary_frame(
        self, data: bytes, recv_ts: float
    ) -> FrameRecord | None:
        if len(data) < 1:
            self.metrics.decode.observe(0)
            return None

        msg_type = data[0]

        if msg_type == 0x01:
            # MessagePack: {"ts": float, "lms": [543 floats]}
            msg = msgpack.unpackb(data[1:], raw=False)
            ts = float(msg.get("ts", 0.0))
            lms_list = msg.get("lms", [])
            if len(lms_list) != LANDMARK_DIM:
                return None
            landmarks = np.asarray(lms_list, dtype=np.float32)
            self.metrics.decode.observe(
                (recv_ts - ts) * 1000 if ts > 0 else 0.0
            )
            return FrameRecord(
                client_ts=ts,
                server_recv_ts=recv_ts,
                landmarks=landmarks,
                seq=self._seq,
            )

        elif msg_type == 0x02:
            # Raw binary: 8-byte float64 ts + 543*4 bytes float32 landmarks
            if len(data) < 9:
                return None
            ts = struct.unpack("<d", data[1:9])[0]
            payload = data[9:]
            if len(payload) != LANDMARK_DIM * 4:
                return None
            landmarks = np.frombuffer(payload, dtype=np.float32)
            self.metrics.decode.observe(
                (recv_ts - ts) * 1000 if ts > 0 else 0.0
            )
            return FrameRecord(
                client_ts=ts,
                server_recv_ts=recv_ts,
                landmarks=landmarks.copy(),
                seq=self._seq,
            )

        elif msg_type == 0x00:
            # Legacy JSON for backward compatibility
            text = data[1:].decode("utf-8")
            msg = json.loads(text)
            ts = float(msg.get("ts", 0.0))
            lms_list = msg.get("lms", [])
            if len(lms_list) != LANDMARK_DIM:
                return None
            landmarks = np.asarray(lms_list, dtype=np.float32)
            self.metrics.decode.observe(
                (recv_ts - ts) * 1000 if ts > 0 else 0.0
            )
            return FrameRecord(
                client_ts=ts,
                server_recv_ts=recv_ts,
                landmarks=landmarks,
                seq=self._seq,
            )

        return None

    def _enqueue_frame(self, frame: FrameRecord) -> None:
        self._seq += 1
        if len(self.frame_queue) >= MAX_QUEUE_SIZE:
            self.frame_queue.popleft()
        self.frame_queue.append(frame)

    async def _run_inference(self, frame: FrameRecord) -> None:
        infer_start = time.perf_counter()

        # Check if we're behind — frame dropping policy
        behind_ms = (time.perf_counter() - frame.server_recv_ts) * 1000
        if behind_ms > LATENCY_HORIZON:
            return

        # Dummy prediction using landmarks (placeholder for ONNX model)
        prediction = self._dummy_predict(frame.landmarks)
        infer_ms = (time.perf_counter() - infer_start) * 1000
        self.metrics.infer.observe(infer_ms)

        latency_ms = (time.perf_counter() - frame.client_ts) * 1000
        await self._send_result(prediction, latency_ms, frame.seq)

    def _dummy_predict(self, landmarks: np.ndarray) -> dict[str, Any]:
        """Placeholder inference result - to be replaced with ONNX Runtime."""
        norm = float(np.linalg.norm(landmarks))
        category = "gesture_A" if norm < 10.0 else "gesture_B"
        return {
            "category": category,
            "confidence": round(float(np.mean(landmarks)) * 0.1, 4),
            "timestamp": time.time(),
            "landmark_norm": round(norm, 4),
        }

    async def _send_result(
        self, prediction: dict, total_latency_ms: float, seq: int
    ) -> None:
        payload = msgpack.packb(
            {
                "seq": seq,
                "latency_ms": round(total_latency_ms, 2),
                "prediction": prediction,
            },
            use_bin_type=True,
        )
        message = b"\x03" + payload
        await self.websocket.send_bytes(message)


class ConnectionManager:
    """Tracks active WebSocket connections and their metrics."""

    def __init__(self) -> None:
        self.active: list[RecognitionConnection] = []

    def connect(self, conn: RecognitionConnection) -> None:
        self.active.append(conn)

    def disconnect(self, conn: RecognitionConnection) -> None:
        self.active.remove(conn)

    def metrics_snapshot(self) -> dict[str, dict[str, float]]:
        if not self.active:
            return {}
        return self.active[-1].metrics.snapshot()


_manager = ConnectionManager()


@router.websocket("/ws/landmarks")
async def landmark_websocket(websocket: WebSocket) -> None:
    conn = RecognitionConnection(websocket)
    _manager.connect(conn)
    await conn.handle()
    _manager.disconnect(conn)


@router.get("/ws/metrics")
async def get_metrics() -> dict[str, dict[str, float]]:
    """Return current latency metrics snapshot from the most recent connection."""
    return _manager.metrics_snapshot()
