import os
import mujoco
import numpy as np
from logger_config import get_logger

class CameraRenderer:
    """
    Robust head-less off-screen renderer for MuJoCo cameras with better error handling.
    """

    def __init__(self, model, data, camera_name=None, width=640, height=480):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.logger = get_logger('CameraRenderer')
        
        # Initialize to None - will be created on first use
        self.gl_ctx = None
        self.mjr_ctx = None
        self.opt = None
        self.scn = None
        self.viewport = None
        self.cam = None
        self.rgb = None
        self.initialized = False
        
        # Cache available cameras
        self.available_cameras = self._get_available_cameras()
        self.logger.info(f"Available cameras: {list(self.available_cameras.keys())}")

    def _get_available_cameras(self):
        """Get all available cameras in the model"""
        cameras = {}
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                cameras[name] = i
        return cameras

    def _lazy_init(self):
        """Initialize OpenGL context only when needed"""
        if self.initialized:
            return True
            
        try:
            # Try to create OpenGL context
            self.gl_ctx = mujoco.GLContext(self.width, self.height)
            self.gl_ctx.make_current()
            
            # Create MuJoCo rendering resources
            self.mjr_ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.opt = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.opt)
            self.scn = mujoco.MjvScene(self.model, maxgeom=10_000)
            self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
            
            # Pre-allocate RGB buffer
            self.rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
            
            # Create camera
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            
            self.initialized = True
            self.logger.info("Camera renderer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera renderer: {e}")
            self.initialized = False
            return False

    def set_camera(self, name):
        """Set camera by name with validation"""
        if not self._lazy_init():
            raise RuntimeError("Camera renderer not initialized")
            
        if name not in self.available_cameras:
            available = list(self.available_cameras.keys())
            raise ValueError(f"Camera '{name}' not found. Available: {available}")
        
        try:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.available_cameras[name]
            return True
        except Exception as e:
            self.logger.error(f"Failed to set camera '{name}': {e}")
            return False

    def get_rgb_image(self):
        """Get RGB image with error handling"""
        if not self.initialized:
            if not self._lazy_init():
                self.logger.error("Cannot get image - renderer not initialized")
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        try:
            # Update scene
            mujoco.mjv_updateScene(
                self.model, self.data, self.opt, None, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL, self.scn
            )
            
            # Render to offscreen buffer
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mjr_ctx)
            mujoco.mjr_render(self.viewport, self.scn, self.mjr_ctx)
            mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.mjr_ctx)
            
            return self.rgb.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to render image: {e}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.gl_ctx:
                # Note: GLContext cleanup is automatic in newer MuJoCo versions
                pass
            self.initialized = False
            self.logger.info("Camera renderer cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")