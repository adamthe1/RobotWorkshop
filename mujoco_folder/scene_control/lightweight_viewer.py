from __future__ import annotations

import glfw
import mujoco
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from PIL import Image
import time


class LightweightViewer:
    """
    Minimal GLFW-based MuJoCo viewer.

    Exposes a small API compatible with mujoco.viewer passive mode:
    - is_running(): whether the window is open
    - sync(): render current model/data
    - close(): destroy the window and context
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 width: int = 800, height: int = 600, title: str = "MuJoCo Viewer"):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.cam = None
        self.opt = None
        self.scn = None
        self.ctx = None
        # Input state
        self._last_x = None
        self._last_y = None
        self._left_down = False
        self._right_down = False
        # Recording state
        self._recording: bool = False
        self._last_capture_time: Optional[float] = None
        self._frame_index: int = 0
        # Recording controls
        self._record_stride: int = 2  # capture every 1/X frames; X is stride
        self._record_scale: float = 0.4  # downscale factor (e.g., 0.5)
        # Output directory for frames (set on start)
        self._frames_dir: Optional[str] = None
        self._exit_requested = False

    def launch(self) -> "LightweightViewer":
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(self.width, self.height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

        # Initialize default camera and options
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        # Set a sensible camera to avoid black screen
        center = self.model.stat.center
        extent = float(self.model.stat.extent)
        self.cam.lookat = center
        self.cam.distance = 2.5 * extent if extent > 0 else 2.5
        self.cam.azimuth = 90
        self.cam.elevation = -20

        # Set input callbacks
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_key_callback(self.window, self._on_key)  

        return self
    

    def _on_key(self, window, key, scancode, action, mods):
        """Handle keyboard input"""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            self._exit_requested = True
        if key == glfw.KEY_R and action == glfw.PRESS:
            if not self._recording:
                self._start_recording()
            else:
                self._stop_recording()


    def is_running(self) -> bool:
        return self.window is not None and not glfw.window_should_close(self.window)

    def sync(self) -> None:
        if not self.is_running():
            return

        # Update scene from current state
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, fb_w, fb_h)
        mujoco.mjr_render(viewport, self.scn, self.ctx)

        # If recording, capture the rendered frame before buffer swap
        if self._recording:
            self._frame_index += 1
            if self._record_stride <= 1 or (self._frame_index % self._record_stride == 0):
                now = time.time()
                if self._last_capture_time is None:
                    self._last_capture_time = now
                # Save frame to disk with timestamp
                self._capture_and_save_frame(viewport, now)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # -------------------- Callbacks & Controls --------------------
    def _on_resize(self, window, width, height):
        # Nothing special needed; rendering uses framebuffer size each frame
        pass

    def _on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_down = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_down = action == glfw.PRESS
        # Reset last to avoid jump on next move
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.window)
            self._last_x, self._last_y = x, y

    def _on_cursor_pos(self, window, x, y):
        if self._last_x is None or self._last_y is None:
            self._last_x, self._last_y = x, y
            return
        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x, self._last_y = x, y

        if not (self._left_down or self._right_down):
            return

        # Normalize deltas to a reasonable scale
        scale = 0.003
        if self._left_down:
            # Rotate: horizontal + vertical
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                scale * dx,
                scale * dy,
                self.scn,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                scale * dx,
                scale * dy,
                self.scn,
                self.cam,
            )
        elif self._right_down:
            # Pan: horizontal + vertical translation
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_H,
                scale * dx,
                scale * dy,
                self.scn,
                self.cam,
            )
            mujoco.mjv_moveCamera(
                self.model,
                mujoco.mjtMouse.mjMOUSE_MOVE_V,
                scale * dx,
                scale * dy,
                self.scn,
                self.cam,
            )

    def _on_scroll(self, window, xoffset, yoffset):
        # Zoom in/out with scroll
        scale = 0.05
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -scale * yoffset,
            self.scn,
            self.cam,
        )


    def close(self) -> None:
        glfw.terminate()

    # -------------------- Recording Helpers --------------------
    def _start_recording(self) -> None:
        self._recording = True
        self._frame_index = 0
        self._last_capture_time = None

        base_dir = os.environ.get("MAIN_DIRECTORY", os.getcwd())
        gifs_dir = os.path.join(base_dir, "example_gifs")
        images_dir = os.path.join(gifs_dir, "imagesforgif")
        os.makedirs(images_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a session subfolder to avoid mixing recordings
        self._frames_dir = os.path.join(images_dir, f"session_{timestamp}")
        os.makedirs(self._frames_dir, exist_ok=True)

    def _stop_recording(self) -> None:
        self._recording = False
        self._last_capture_time = None
        self._frames_dir = None

    def _capture_and_save_frame(self, viewport: mujoco.MjrRect, timestamp_sec: float) -> None:
        if self._frames_dir is None:
            return
        fb_w, fb_h = viewport.width, viewport.height
        if fb_w <= 0 or fb_h <= 0:
            return
        # Allocate arrays for rgb and depth as required by mjr_readPixels
        rgb = np.empty((fb_h, fb_w, 3), dtype=np.uint8)
        depth = np.empty((fb_h, fb_w), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth, viewport, self.ctx)
        # Flip vertically to match conventional image coordinates
        rgb = np.flipud(rgb)
        # Optional downscale using PIL if requested
        if self._record_scale and self._record_scale > 0 and self._record_scale != 1.0:
            img = Image.fromarray(rgb, mode="RGB")
            new_w = max(1, int(img.width * self._record_scale))
            new_h = max(1, int(img.height * self._record_scale))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            rgb = np.asarray(img)
        # Save to disk as PNG with index and timestamp in filename
        ms = int(timestamp_sec * 1000)
        filename = f"frame_{self._frame_index:06d}_{ms}.png"
        out_path = os.path.join(self._frames_dir, filename)
        Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG", optimize=True)

    # -------------------- Public setters --------------------
    def set_record_stride(self, stride: int) -> None:
        """Set capture stride (record 1 frame every `stride` frames)."""
        self._record_stride = max(1, int(stride))

    def set_record_scale(self, scale: float) -> None:
        """Set downscale factor for recording, e.g., 0.5 to halve resolution."""
        self._record_scale = float(scale) if scale > 0 else 1.0
    
    def exit_requested(self) -> bool:
        """Return True if user requested exit (e.g., pressed ESC)."""
        return self._exit_requested
