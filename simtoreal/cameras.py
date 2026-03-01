"""
cameras.py — Camera abstraction for real Franka Panda deployment.

Must match the training camera configuration exactly:
    CAMERA_KEYS = ("front", "wrist", "left_shoulder", "right_shoulder")
These 4 views are positionally encoded by the CNN encoder — order matters.

Supports:
  - Orbbec depth cameras (Gemini, Femto, Astra …) via pyorbbecsdk
  - Intel RealSense cameras (D435 / D415 …) via pyrealsense2
  - USB webcams via OpenCV
  - Dummy cameras that return zero frames (for missing viewpoints)
  - Any mix: specify which hardware you have, the rest auto-fills with Dummy

Each camera returns (3, H, W) uint8 RGB numpy arrays at 84×84 by default.
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Canonical camera keys — MUST match RLBench training config exactly.
# Defined in: cfgs/rlbench_task/default.yaml
#   camera_keys: [front, wrist, left_shoulder, right_shoulder]
# And in: rlbench_src/rlbench_env.py constructor default.
#
# The CNN encoder (MultiViewCNNEncoder) processes these views positionally:
#   index 0 = front
#   index 1 = wrist
#   index 2 = left_shoulder
#   index 3 = right_shoulder
# ---------------------------------------------------------------------------
CAMERA_KEYS: tuple[str, ...] = ("front", "wrist", "left_shoulder", "right_shoulder")
NUM_CAMERAS: int = len(CAMERA_KEYS)
CAMERA_H: int = 84
CAMERA_W: int = 84


# ============================================================================
# Base class
# ============================================================================

class CameraBase(abc.ABC):
    """Abstract camera that returns a (3, H, W) uint8 RGB frame."""

    def __init__(self, name: str, height: int = 84, width: int = 84):
        self.name = name
        self.height = height
        self.width = width

    @abc.abstractmethod
    def capture(self) -> np.ndarray:
        """Return (3, H, W) uint8 RGB."""
        ...

    def close(self):
        pass


# ============================================================================
# Dummy camera – returns black frames
# ============================================================================

class DummyCamera(CameraBase):
    """Returns all-zero frames.  Use for camera slots you don't have hardware for."""

    def capture(self) -> np.ndarray:
        return np.zeros((3, self.height, self.width), dtype=np.uint8)


# ============================================================================
# Intel RealSense camera (RGB stream)
# ============================================================================

class RealSenseCamera(CameraBase):
    """
    Wraps a single Intel RealSense device (D435, D415, etc.).

    Parameters
    ----------
    name : str
        Logical name used by the env (e.g. "wrist", "front").
    serial : str or None
        RealSense serial number.  None = use the first device found.
    height, width : int
        Desired output resolution (the captured frame will be resized).
    fps : int
        Capture framerate.
    stream_type : str
        "rgb" for colour-only, "rgbd" to also enable depth (depth not
        returned by capture() but accessible via capture_depth()).
    """

    def __init__(
        self,
        name: str,
        serial: Optional[str] = None,
        height: int = 84,
        width: int = 84,
        fps: int = 30,
        stream_type: str = "rgb",
    ):
        super().__init__(name, height, width)
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 is required for RealSense cameras.  "
                "Install with: pip install pyrealsense2"
            )
        self._rs = rs
        self._serial = serial
        self._stream_type = stream_type

        cfg = rs.config()
        if serial is not None:
            cfg.enable_device(serial)
        # Enable colour stream at a reasonable native resolution; we resize later
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        if stream_type == "rgbd":
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(cfg)
        # Let auto-exposure settle
        for _ in range(30):
            self._pipeline.wait_for_frames()
        print(f"[Camera] RealSense '{name}' ready (serial={serial})")

    def capture(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        colour = np.asanyarray(frames.get_color_frame().get_data())  # (H, W, 3) BGR
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        colour = cv2.resize(colour, (self.width, self.height))
        return colour.transpose(2, 0, 1).copy()  # (3, H, W) uint8

    def capture_depth(self) -> Optional[np.ndarray]:
        """Return (H, W) uint16 depth map, or None if depth not enabled."""
        if self._stream_type != "rgbd":
            return None
        frames = self._pipeline.wait_for_frames()
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        depth = cv2.resize(depth, (self.width, self.height))
        return depth

    def close(self):
        self._pipeline.stop()


# ============================================================================
# USB / laptop webcam (for cheap external views)
# ============================================================================

class USBCamera(CameraBase):
    """OpenCV VideoCapture camera (USB webcam, laptop cam, etc.)."""

    def __init__(
        self,
        name: str,
        device_id: int = 0,
        height: int = 84,
        width: int = 84,
    ):
        super().__init__(name, height, width)
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {device_id}")
        # Set a higher native resolution for cleaner downscale
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Warm-up frames (auto-exposure settle)
        for _ in range(10):
            self._cap.read()
        print(f"[Camera] USB '{name}' ready (device_id={device_id})")

    def capture(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            return np.zeros((3, self.height, self.width), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.width, self.height))
        return frame.transpose(2, 0, 1).copy()

    def close(self):
        self._cap.release()


# ============================================================================
# Orbbec depth camera (Gemini 335, Femto, Astra Stereo, etc.)
# ============================================================================

class OrbbecCamera(CameraBase):
    """
    Orbbec stereo/depth camera — RGB stream only (CQN-AS doesn't need depth).

    Uses the pyorbbecsdk (Orbbec SDK v2 Python bindings).
    Install:
        pip install pyorbbecsdk
    or build from source: https://github.com/orbbec/pyorbbecsdk

    Parameters
    ----------
    name : str
        Logical camera name (e.g. "front", "wrist").
    serial : str or None
        Device serial number.  If None and only one Orbbec camera is
        connected, that device is used automatically.
    height, width : int
        Output resolution (captured frame is resized).
    fps : int
        Desired colour stream frame-rate.
    """

    def __init__(
        self,
        name: str,
        serial: Optional[str] = None,
        height: int = 84,
        width: int = 84,
        fps: int = 30,
    ):
        super().__init__(name, height, width)
        try:
            from pyorbbecsdk import (
                Config,
                Context,
                OBFormat,
                OBSensorType,
                Pipeline,
            )
        except ImportError:
            raise ImportError(
                "pyorbbecsdk is required for Orbbec cameras.  "
                "Install with:  pip install pyorbbecsdk  "
                "or build from source: https://github.com/orbbec/pyorbbecsdk"
            )
        self._ob = {
            "Config": Config,
            "Context": Context,
            "OBFormat": OBFormat,
            "OBSensorType": OBSensorType,
            "Pipeline": Pipeline,
        }
        self._serial = serial
        self._fps = fps
        self._pipeline = None

        # Resolve device
        ctx = Context()
        device_list = ctx.query_devices()
        n = device_list.get_count()
        if n == 0:
            raise RuntimeError("No Orbbec devices found")

        device = None
        if serial is not None:
            for i in range(n):
                dev = device_list.get_device_by_index(i)
                info = dev.get_device_info()
                if info.get_serial_number() == serial:
                    device = dev
                    break
            if device is None:
                available = []
                for i in range(n):
                    dev = device_list.get_device_by_index(i)
                    available.append(dev.get_device_info().get_serial_number())
                raise RuntimeError(
                    f"Orbbec device with serial '{serial}' not found.  "
                    f"Available: {available}"
                )
        else:
            # Auto-select first device
            device = device_list.get_device_by_index(0)
            found_serial = device.get_device_info().get_serial_number()
            print(f"[OrbbecCamera] '{name}': auto-selected device serial={found_serial}")

        # Configure colour-only pipeline
        self._pipeline = Pipeline(device)
        config = Config()
        try:
            profile_list = self._pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            # Try to get 640×480 RGB at requested fps
            try:
                color_profile = profile_list.get_video_stream_profile(
                    640, 0, OBFormat.RGB, fps
                )
            except Exception:
                # Fall back to default profile
                color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
        except Exception as e:
            raise RuntimeError(
                f"Orbbec camera '{name}' (serial={serial}): "
                f"failed to configure colour stream: {e}"
            )

        self._pipeline.start(config)
        # Warm-up: let auto-exposure settle
        for _ in range(30):
            self._pipeline.wait_for_frames(200)
        print(
            f"[Camera] Orbbec '{name}' ready "
            f"(serial={serial or 'auto'}, profile={color_profile})"
        )

    def capture(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames(200)
        if frames is None:
            return np.zeros((3, self.height, self.width), dtype=np.uint8)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return np.zeros((3, self.height, self.width), dtype=np.uint8)

        # Convert to numpy BGR image
        width = color_frame.get_width()
        height = color_frame.get_height()
        data = np.asanyarray(color_frame.get_data())
        fmt = color_frame.get_format()

        from pyorbbecsdk import OBFormat

        if fmt == OBFormat.RGB:
            image = data.reshape((height, width, 3))  # already RGB
        elif fmt == OBFormat.BGR:
            image = data.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif fmt == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif fmt == OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUYV)
        elif fmt == OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB_UYVY)
        else:
            print(f"[OrbbecCamera] Unsupported format {fmt}, returning zeros")
            return np.zeros((3, self.height, self.width), dtype=np.uint8)

        # Resize to target
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return image.transpose(2, 0, 1).copy()  # (3, H, W) uint8 RGB

    def close(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


# ============================================================================
# Multi-camera rig
# ============================================================================

class CameraRig:
    """
    Manages the set of cameras expected by CQN-AS.

    The default RLBench training uses 4 views:
        ("front", "wrist", "left_shoulder", "right_shoulder")

    For real deployment you might only have a wrist RealSense and fill the
    other slots with DummyCamera.

    Parameters
    ----------
    cameras : dict[str, CameraBase]
        Mapping from camera_key → camera instance.
    camera_keys : tuple[str]
        Ordered list of keys matching the training config (order matters
        because the CNN encoder processes views positionally).
    """

    def __init__(
        self,
        cameras: dict[str, CameraBase],
        camera_keys: tuple[str, ...] = CAMERA_KEYS,
    ):
        self.camera_keys = camera_keys
        self.cameras: dict[str, CameraBase] = {}
        for key in camera_keys:
            if key in cameras:
                self.cameras[key] = cameras[key]
            else:
                print(f"[CameraRig] No camera for '{key}' — using DummyCamera")
                h = next(iter(cameras.values())).height if cameras else 84
                w = next(iter(cameras.values())).width if cameras else 84
                self.cameras[key] = DummyCamera(key, h, w)

    def capture_all(self) -> np.ndarray:
        """Return (V, 3, H, W) uint8 array of all views in order."""
        return np.stack([self.cameras[k].capture() for k in self.camera_keys], axis=0)

    def close(self):
        for cam in self.cameras.values():
            cam.close()


# ============================================================================
# Convenience constructors
# ============================================================================

def make_wrist_only_rig(
    serial: Optional[str] = None,
    height: int = CAMERA_H,
    width: int = CAMERA_W,
    camera_keys: tuple[str, ...] = CAMERA_KEYS,
) -> CameraRig:
    """Create a rig with only a wrist RealSense; other views are zero-filled."""
    wrist = RealSenseCamera("wrist", serial=serial, height=height, width=width,
                            stream_type="rgbd")
    return CameraRig({"wrist": wrist}, camera_keys=camera_keys)


def make_full_rig(
    serials: dict[str, Optional[str]],
    height: int = CAMERA_H,
    width: int = CAMERA_W,
    camera_keys: tuple[str, ...] = CAMERA_KEYS,
) -> CameraRig:
    """
    Create a rig with RealSense cameras for each specified key.

    Parameters
    ----------
    serials : dict
        e.g. {"front": "1234...", "wrist": "5678...", ...}
        Keys not present get DummyCamera automatically.
    """
    cameras = {}
    for key, ser in serials.items():
        cameras[key] = RealSenseCamera(key, serial=ser, height=height, width=width)
    return CameraRig(cameras, camera_keys=camera_keys)


def make_orbbec_rig(
    serials: dict[str, Optional[str]],
    height: int = CAMERA_H,
    width: int = CAMERA_W,
    camera_keys: tuple[str, ...] = CAMERA_KEYS,
) -> CameraRig:
    """
    Create a rig with Orbbec cameras for each specified key.

    Parameters
    ----------
    serials : dict
        Mapping from camera key → Orbbec serial number.
        e.g. {"front": "AB12...", "wrist": "CD34...",
              "left_shoulder": "EF56...", "right_shoulder": "GH78..."}
        Keys not present get DummyCamera automatically.
    """
    cameras = {}
    for key, ser in serials.items():
        cameras[key] = OrbbecCamera(key, serial=ser, height=height, width=width)
    return CameraRig(cameras, camera_keys=camera_keys)


def make_dummy_rig(
    height: int = CAMERA_H,
    width: int = CAMERA_W,
    camera_keys: tuple[str, ...] = CAMERA_KEYS,
) -> CameraRig:
    """All dummy cameras — useful for testing without any camera hardware."""
    return CameraRig({}, camera_keys=camera_keys)
