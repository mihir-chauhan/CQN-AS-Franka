"""
test.py — Live image stream viewer for all connected Orbbec cameras.

Opens one OpenCV window per camera. Press 'q' to quit.
"""

import cv2
import numpy as np
from pyorbbecsdk import Config, Context, OBFormat, OBSensorType, Pipeline


def discover_devices():
    """Return list of (index, serial, name) for all connected Orbbec devices."""
    ctx = Context()
    device_list = ctx.query_devices()
    devices = []
    for i in range(device_list.get_count()):
        device = device_list.get_device_by_index(i)
        info = device.get_device_info()
        devices.append((i, info.get_serial_number(), info.get_name()))
    return ctx, device_list, devices


def start_pipeline(device, fps=30):
    """Start a colour-only pipeline on the given device. Returns (pipeline, profile)."""
    pipeline = Pipeline(device)
    config = Config()
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, fps)
    except Exception:
        color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    pipeline.start(config)
    # Warm-up frames to let auto-exposure settle
    for _ in range(15):
        pipeline.wait_for_frames(200)
    return pipeline, color_profile


def grab_rgb(pipeline):
    """Grab one RGB frame from pipeline. Returns HxWx3 uint8 (BGR for OpenCV) or None."""
    frames = pipeline.wait_for_frames(200)
    if frames is None:
        return None
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None

    width = color_frame.get_width()
    height = color_frame.get_height()
    data = np.asanyarray(color_frame.get_data())
    fmt = color_frame.get_format()

    if fmt == OBFormat.RGB:
        image = data.reshape((height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV expects BGR
    elif fmt == OBFormat.BGR:
        image = data.reshape((height, width, 3))
    elif fmt == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.YUYV:
        image = data.reshape((height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.UYVY:
        image = data.reshape((height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print(f"Unsupported format: {fmt}")
        return None

    return image


def main():
    ctx, device_list, devices = discover_devices()
    n = len(devices)

    if n == 0:
        print("No Orbbec cameras found!")
        return

    print(f"Found {n} Orbbec camera(s):")
    for idx, serial, name in devices:
        print(f"  [{idx}] serial={serial}  name={name}")

    # Start a pipeline for each device
    pipelines = []
    window_names = []
    for idx, serial, name in devices:
        device = device_list.get_device_by_index(idx)
        print(f"Starting pipeline for device {idx} (serial={serial})...")
        pipeline, profile = start_pipeline(device)
        pipelines.append(pipeline)
        win = f"Orbbec {idx}: {serial} ({name})"
        window_names.append(win)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)

    print(f"\nStreaming from {n} camera(s). Press 'q' in any window to quit.\n")

    try:
        while True:
            for i, pipeline in enumerate(pipelines):
                frame = grab_rgb(pipeline)
                if frame is not None:
                    cv2.imshow(window_names[i], frame)
                else:
                    # Show a black frame with "No Frame" text
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "No Frame", (220, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    cv2.imshow(window_names[i], blank)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()