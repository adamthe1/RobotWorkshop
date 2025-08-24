import os
import mujoco
import numpy as np


class CameraRenderer:
    """
    Head-less off-screen renderer for any named MuJoCo camera.

    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData            (kept in sync by caller)
    camera_name : str | None         (None → default free camera)
    width, height : int              output resolution
    """

    def __init__(
        self,
        model,
        data,
        camera_name: str | None = None,
        width: int = 640,
        height: int = 480,
    ):
        self.model, self.data = model, data
        self.width, self.height = width, height

        # ---------- 1. OpenGL context (invisible) ---------------------------
        #
        # If the code runs on a GPU machine:
        #   $ export MUJOCO_GL=egl
        # Or on a pure-CPU box:
        #   $ export MUJOCO_GL=osmesa
        #
        # Then the next two lines succeed even with no X-server.
        #
        self.gl_ctx = mujoco.GLContext(width, height)
        self.gl_ctx.make_current()

        # ---------- 2. MuJoCo rendering resources --------------------------
        self.mjr_ctx = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150
        )

        self.opt = mujoco.MjvOption()      # visualisation flags
        mujoco.mjv_defaultOption(self.opt)

        self.scn = mujoco.MjvScene(model, maxgeom=10_000)
        self.viewport = mujoco.MjrRect(0, 0, width, height)
        self.rgb = np.empty((height, width, 3), dtype=np.uint8)

        # ---------- 3. Camera ----------------------------------------------
        self.cam = mujoco.MjvCamera()
        self.set_camera(camera_name) if camera_name else mujoco.mjv_defaultCamera(
            self.cam
        )

    # ------------------------------------------------------------------ #
    # public: switch cameras on the fly
    # ------------------------------------------------------------------ #
    def set_camera(self, name: str):
        cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, name
        )
        if cam_id == -1:
            raise ValueError(f"Camera '{name}' not found in model.")
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

    # ------------------------------------------------------------------ #
    # public: grab one RGB frame (H × W × 3, uint8)
    # ------------------------------------------------------------------ #
    def get_rgb_image(self) -> np.ndarray:
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mjr_ctx
        )
        mujoco.mjr_render(self.viewport, self.scn, self.mjr_ctx)
        mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.mjr_ctx)
        return self.rgb.copy()  # decouple from MuJoCo’s buffer
