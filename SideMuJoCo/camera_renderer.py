import mujoco
import numpy as np

class CameraRenderer:
    def __init__(self, model, data, camera_name=None, width=640, height=480):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.camera_name = camera_name

        # Viewer options
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)

        # Set up rendering context and scene
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scn = mujoco.MjvScene(model, maxgeom=1000)
        self.rgb_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)

        # Set camera
        self.cam = mujoco.MjvCamera()
        if self.camera_name:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
            if cam_id == -1:
                raise ValueError(f"Camera '{self.camera_name}' not found in the model.")
            mujoco.mjv_cameraFromFixed(self.cam, model, data, cam_id)
        else:
            mujoco.mjv_defaultCamera(self.cam)

    def get_rgb_image(self):
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        mujoco.mjr_render(
            mujoco.MjrRect(0, 0, self.width, self.height),
            self.scn, self.ctx
        )
        mujoco.mjr_readPixels(
            self.rgb_buffer, None,
            mujoco.MjrRect(0, 0, self.width, self.height),
            self.ctx
        )
        return self.rgb_buffer.copy()
