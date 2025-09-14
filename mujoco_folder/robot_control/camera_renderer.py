import os
import mujoco
import numpy as np
from logger_config import get_logger

class CameraRenderer:
    """
    Robust off-screen renderer for MuJoCo cameras that works with snapshots.
    Similar to PhysicsStateExtractor - uses copied data for thread safety.
    """

    def __init__(self, model, width=640, height=480):
        self.model = model
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
        
        # Cache available cameras at startup
        self.available_cameras = self._get_available_cameras()
        self.logger.info(f"Available cameras: {list(self.available_cameras.keys())}")

        # Current snapshot data for rendering
        self._current_snapshot = None

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
            # Set environment for headless rendering
            os.environ.setdefault('MUJOCO_GL', 'osmesa')
            os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
            
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

    def update_snapshot(self, snapshot):
        """Update the current snapshot for rendering (called by RobotBodyControl)"""
        self._current_snapshot = snapshot

    def get_camera_images(self, camera_names):
        """
        Get RGB images for multiple cameras using current snapshot.
        
        Args:
            camera_names (list): List of camera names to render
            
        Returns:
            dict: Mapping of camera_name -> numpy array (or None if failed)
        """
        if not self._current_snapshot:
            self.logger.warning("No snapshot available for camera rendering")
            return {}

        if not self._lazy_init():
            self.logger.error("Camera renderer not initialized")
            return {}

        images = {}
        
        # Create temporary data from snapshot for rendering
        temp_data = mujoco.MjData(self.model)
        self._copy_snapshot_to_data(temp_data, self._current_snapshot)
        
        for camera_name in camera_names:
            try:
                image = self._render_single_camera(camera_name, temp_data)
                images[camera_name] = image
                if image is not None:
                    self.logger.debug(f"Successfully rendered {camera_name}")
                else:
                    self.logger.warning(f"Failed to render {camera_name}")
            except Exception as e:
                self.logger.error(f"Error rendering camera {camera_name}: {e}")
                images[camera_name] = None
                
        return images

    def _copy_snapshot_to_data(self, temp_data, snapshot):
        """Copy snapshot data to temporary MjData for rendering"""
        try:
            # Copy essential state for rendering
            temp_data.qpos[:] = snapshot.qpos
            temp_data.qvel[:] = snapshot.qvel
            temp_data.ctrl[:] = snapshot.ctrl
            temp_data.time = snapshot.time
            
            # Copy body poses (important for camera positioning)
            temp_data.xpos[:] = snapshot.xpos
            temp_data.xquat[:] = snapshot.xquat
            temp_data.xmat[:] = snapshot.xmat
            
            # Copy site positions (for camera attachments)
            temp_data.site_xpos[:] = snapshot.site_xpos
            temp_data.site_xmat[:] = snapshot.site_xmat
            
            # Forward kinematics to update dependent quantities
            mujoco.mj_forward(self.model, temp_data)
            
        except Exception as e:
            self.logger.error(f"Failed to copy snapshot to temp data: {e}")
            raise

    def _render_single_camera(self, camera_name, temp_data):
        """Render a single camera using temporary data"""
        if camera_name not in self.available_cameras:
            available = list(self.available_cameras.keys())
            self.logger.warning(f"Camera '{camera_name}' not found. Available: {available}")
            return None
        
        try:
            # Set camera
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = self.available_cameras[camera_name]
            
            # Update scene with temporary data
            mujoco.mjv_updateScene(
                self.model, temp_data, self.opt, None, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL, self.scn
            )
            
            # Render to offscreen buffer
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.mjr_ctx)
            mujoco.mjr_render(self.viewport, self.scn, self.mjr_ctx)
            mujoco.mjr_readPixels(self.rgb, None, self.viewport, self.mjr_ctx)
            
            return self.rgb.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to render camera '{camera_name}': {e}")
            return None

    def get_available_camera_list(self):
        """Get list of available camera names"""
        return list(self.available_cameras.keys())

    def validate_camera_names(self, camera_names):
        """
        Validate that camera names exist in the model.
        
        Returns:
            tuple: (valid_cameras, invalid_cameras)
        """
        valid = []
        invalid = []
        
        for name in camera_names:
            if name in self.available_cameras:
                valid.append(name)
            else:
                invalid.append(name)
                
        return valid, invalid

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.gl_ctx:
                # Note: GLContext cleanup is automatic in newer MuJoCo versions
                pass
            self.initialized = False
            self._current_snapshot = None
            self.logger.info("Camera renderer cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")