#!/usr/bin/env python3
# filepath: /home/adam/Documents/coding/autonomous/finetuning/recording.py

import time
import numpy as np
import mujoco
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass
from datetime import datetime
import tempfile
import shutil
import glfw
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class RecordingConfig:
    """Configuration for LeRobot recording following v2.1 format"""
    dataset_name: str = "mujoco_teleop"
    output_dir: str = os.getenv("MAIN_DIRECTORY") + "/finetuning/datasets" 
    fps: int = 30
    robot_type: str = "franka_panda"
    record_video: bool = True
    video_width: int = 224
    video_height: int = 224
    video_fps: int = 30
    camera_names: List[str] = None
    use_robot_prefix: bool = False  # NEW: control robot prefix usage
    robot_prefix: str = "r1_"      # NEW: default robot prefix
    task: str = "robot_manipulation"

    def __post_init__(self):
        if self.camera_names is None:
            # Default camera names from mjx_panda.xml
            base_cameras = ["main_cam", "wrist_cam"]  # Camera in the hand
            
            if self.use_robot_prefix:
                # Add robot prefix to camera names
                self.camera_names = [f"{self.robot_prefix}{cam}" for cam in base_cameras]
            else:
                self.camera_names = base_cameras

class LeRobotDatasetRecorder:
    """Records MuJoCo simulation data in LeRobot v2.1 format"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: RecordingConfig):
        self.model = model
        self.data = data
        self.config = config
        
        # Validate camera names exist in model
        self._validate_cameras()
        
        # Recording state
        self.is_recording = False
        self.current_episode_data = []
        self.current_episode_index = 0
        self.current_frame_index = 0
        self.global_frame_index = 0
        self.episode_start_time = 0.0
        
        # Delete confirmation state
        self.delete_confirmation_pending = False
        self.delete_confirmation_time = 0.0
        
        # Dataset structure
        self.dataset_path = Path(config.output_dir) / config.dataset_name
        self.data_path = self.dataset_path / "data"
        self.videos_path = self.dataset_path / "videos"
        self.meta_path = self.dataset_path / "meta"
        
        # Check if dataset already exists and load existing data
        self._load_existing_dataset()
        
        # Create directory structure
        self._setup_directories()
        
        # Video recording setup
        self.video_writers = {}
        self.temp_video_paths = {}
        
        # Setup cameras for rendering
        if config.record_video:
            self._setup_cameras()
        
        # Dataset metadata (will be loaded from existing if available)
        if not hasattr(self, 'total_frames'):
            self.total_frames = 0
            self.total_episodes = 0
            self.episodes_info = []
            self.episode_stats = []

        self.current_task = config.task
        print(f"[LeRobotRecorder] Initialized v2.1 dataset at: {self.dataset_path}")
        print(f"[LeRobotRecorder] Recording from cameras: {self.config.camera_names}")
    
    def _load_existing_dataset(self):
        """Load existing dataset metadata if it exists"""
        if not self.dataset_path.exists():
            print(f"[LeRobotRecorder] Creating new dataset")
            return
        
        info_path = self.meta_path / 'info.json'
        if info_path.exists():
            print(f"[LeRobotRecorder] Found existing dataset, loading metadata...")
            
            # Load info.json
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.total_episodes = info.get('total_episodes', 0)
                self.total_frames = info.get('total_frames', 0)
                self.current_episode_index = self.total_episodes
                self.global_frame_index = self.total_frames
            
            # Load episodes.jsonl
            episodes_path = self.meta_path / 'episodes.jsonl'
            self.episodes_info = []
            if episodes_path.exists():
                with open(episodes_path, 'r') as f:
                    for line in f:
                        self.episodes_info.append(json.loads(line.strip()))
            
            # Load episodes_stats.jsonl
            stats_path = self.meta_path / 'episodes_stats.jsonl'
            self.episode_stats = []
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    for line in f:
                        self.episode_stats.append(json.loads(line.strip()))
            
            print(f"[LeRobotRecorder] Loaded existing dataset with {self.total_episodes} episodes, {self.total_frames} frames")
        else:
            print(f"[LeRobotRecorder] Creating new dataset")
    
    def _validate_cameras(self):
        """Validate that specified cameras exist in the model"""
        available_cameras = []
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name:
                available_cameras.append(cam_name)
        
        print(f"[LeRobotRecorder] Available cameras in model: {available_cameras}")
        
        # Check if requested cameras exist
        missing_cameras = []
        for cam_name in self.config.camera_names:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                missing_cameras.append(cam_name)
        
        if missing_cameras:
            print(f"[LeRobotRecorder] WARNING: Missing cameras: {missing_cameras}")
            print(f"[LeRobotRecorder] Will use free cameras for missing ones")
    
    def _setup_directories(self):
        """Create LeRobot dataset directory structure"""
        # Main directories
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
        self.videos_path.mkdir(exist_ok=True)
        self.meta_path.mkdir(exist_ok=True)
        
        # Chunk directories (using chunk-000 for simplicity)
        self.data_chunk_path = self.data_path / "chunk-000"
        self.data_chunk_path.mkdir(exist_ok=True)
        
        if self.config.record_video:
            self.video_chunk_path = self.videos_path / "chunk-000"
            self.video_chunk_path.mkdir(exist_ok=True)
            for cam_name in self.config.camera_names:
                (self.video_chunk_path / cam_name).mkdir(exist_ok=True)
    
    def _setup_cameras(self):
        """Setup camera rendering contexts"""
        self.cameras = {}
        self.renderers = {}
        
        for cam_name in self.config.camera_names:
            # Try to find camera by name
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            
            camera = mujoco.MjvCamera()
            if cam_id >= 0:
                camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
                camera.fixedcamid = cam_id
                print(f"[LeRobotRecorder] Using fixed camera '{cam_name}' (id: {cam_id})")
            else:
                # Default free camera setup
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.distance = 2.0
                camera.azimuth = 45.0
                camera.elevation = -30.0
                camera.lookat = np.array([0.0, 0.0, 0.5])
                print(f"[LeRobotRecorder] Using free camera for '{cam_name}' (not found in model)")
            
            self.cameras[cam_name] = camera
            
            # Create renderer
            renderer = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.renderers[cam_name] = renderer

    def start_recording(self):
        """Start recording a new episode"""
        if self.is_recording:
            self.stop_recording()
        
        self.is_recording = True
        self.current_episode_data = []
        self.current_frame_index = 0
        self.episode_start_time = time.time()
        
        # Setup video writers for this episode
        if self.config.record_video:
            self._start_video_recording()
        
        print(f"[LeRobotRecorder] Started recording episode {self.current_episode_index}")
    
    def _start_video_recording(self):
        """Initialize video writers for the current episode"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        for cam_name in self.config.camera_names:
            # Create temporary file for video
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir) / f"episode_{self.current_episode_index:06d}.mp4"
            self.temp_video_paths[cam_name] = temp_path
            
            # Create video writer
            writer = cv2.VideoWriter(
                str(temp_path),
                fourcc,
                self.config.video_fps,
                (self.config.video_width, self.config.video_height)
            )
            self.video_writers[cam_name] = writer
    
    def record_frame(self, action: np.ndarray, done: bool = False, task_index: int = 0):
        """Record a single frame of data"""
        if not self.is_recording:
            return
        
        current_time = time.time()
        timestamp = current_time - self.episode_start_time
        
        # Get robot state (joint positions)
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Create observation state (joint positions for now)
        observation_state = qpos.copy()
        
        # Record images and write to video
        if self.config.record_video:
            for cam_name in self.config.camera_names:
                image = self._render_camera(cam_name)
                # Convert RGB to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.video_writers[cam_name].write(image_bgr)
        
        # Create frame data following LeRobot format
        frame_data = {
            'observation.state': observation_state.tolist(),
            'action': action.tolist(),
            'timestamp': timestamp,
            'episode_index': self.current_episode_index,
            'frame_index': self.current_frame_index,
            'index': self.global_frame_index,
            'task_index': task_index,
        }
        
        # Add done flag if this is the last frame
        if done:
            frame_data['next.done'] = True
        
        self.current_episode_data.append(frame_data)
        self.current_frame_index += 1
        self.global_frame_index += 1
        
        if done:
            self.stop_recording()
    
    def _render_camera(self, cam_name: str) -> np.ndarray:
        """Render image from specified camera"""
        if cam_name not in self.cameras:
            return np.zeros((self.config.video_height, self.config.video_width, 3), dtype=np.uint8)
        
        camera = self.cameras[cam_name]
        renderer = self.renderers[cam_name]
        
        # Create scene and render
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        opt = mujoco.MjvOption()
        
        mujoco.mjv_updateScene(
            self.model, self.data, opt, None, camera,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )
        
        viewport = mujoco.MjrRect(0, 0, self.config.video_width, self.config.video_height)
        mujoco.mjr_render(viewport, scene, renderer)
        
        # Read pixels
        rgb = np.zeros((self.config.video_height, self.config.video_width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, renderer)
        
        # Flip vertically (OpenGL to image convention)
        rgb = np.flipud(rgb)
        
        return rgb
    
    def stop_recording(self):
        """Stop recording and save current episode"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if len(self.current_episode_data) == 0:
            print("[LeRobotRecorder] No frames recorded, skipping episode")
            return
        
        # Finalize videos
        if self.config.record_video:
            self._finalize_videos()
        
        # Save episode data as parquet
        self._save_episode_parquet()
        
        # Update episode metadata
        self._update_episode_metadata()
        
        # Calculate and store episode statistics
        self._calculate_episode_stats()
        
        self.current_episode_index += 1
        self.total_episodes += 1
        self.total_frames += len(self.current_episode_data)
        
        print(f"[LeRobotRecorder] Saved episode {self.current_episode_index - 1} with {len(self.current_episode_data)} frames")
    
    def _finalize_videos(self):
        """Close video writers and move videos to final location"""
        for cam_name in self.config.camera_names:
            if cam_name in self.video_writers:
                # Close writer
                self.video_writers[cam_name].release()
                
                # Move to final location
                final_path = (self.video_chunk_path / cam_name / 
                             f"episode_{self.current_episode_index:06d}.mp4")
                shutil.move(str(self.temp_video_paths[cam_name]), str(final_path))
                
                # Cleanup
                self.temp_video_paths[cam_name].parent.rmdir()
        
        self.video_writers.clear()
        self.temp_video_paths.clear()
    
    def _save_episode_parquet(self):
        """Save episode data as parquet file"""
        # Add image observations if videos were recorded
        if self.config.record_video:
            for frame_data in self.current_episode_data:
                for cam_name in self.config.camera_names:
                    video_path = f"videos/chunk-000/{cam_name}/episode_{self.current_episode_index:06d}.mp4"
                    video_timestamp = frame_data['frame_index'] / self.config.video_fps
                    
                    frame_data[f'observation.images.{cam_name}'] = {
                        'path': video_path,
                        'timestamp': video_timestamp
                    }
        
        # Create DataFrame and save
        df = pd.DataFrame(self.current_episode_data)
        parquet_path = self.data_chunk_path / f"episode_{self.current_episode_index:06d}.parquet"
        df.to_parquet(parquet_path, index=False)
    
    def _update_episode_metadata(self):
        """Update episode metadata"""
        episode_info = {
            'episode_index': self.current_episode_index,
            'length': len(self.current_episode_data),
            'tasks': [self.current_task]
        }
        self.episodes_info.append(episode_info)
    
    def _calculate_episode_stats(self):
        """Calculate statistics for the current episode"""
        df = pd.DataFrame(self.current_episode_data)
        
        stats = {}
        
        # Stats for observation.state
        obs_state = np.array(df['observation.state'].tolist())
        stats['observation.state'] = {
            'max': obs_state.max(axis=0).tolist(),
            'min': obs_state.min(axis=0).tolist(), 
            'mean': obs_state.mean(axis=0).tolist(),
            'std': obs_state.std(axis=0).tolist()
        }
        
        # Stats for action
        action = np.array(df['action'].tolist())
        stats['action'] = {
            'max': action.max(axis=0).tolist(),
            'min': action.min(axis=0).tolist(),
            'mean': action.mean(axis=0).tolist(),
            'std': action.std(axis=0).tolist()
        }
        
        # Stats for timestamp
        timestamps = df['timestamp'].values
        stats['timestamp'] = {
            'max': float(timestamps.max()),
            'min': float(timestamps.min()),
            'mean': float(timestamps.mean()),
            'std': float(timestamps.std())
        }
        
        episode_stats = {
            'episode_index': self.current_episode_index,
            'stats': stats
        }
        self.episode_stats.append(episode_stats)
    
    def delete_previous_recording(self):
        """Delete the most recent episode recording"""
        if self.total_episodes == 0:
            print("[LeRobotRecorder] No episodes to delete")
            return
        
        if self.is_recording:
            print("[LeRobotRecorder] Cannot delete while recording. Stop recording first.")
            return
        
        # Get the most recent episode index
        last_episode_index = self.total_episodes - 1
        
        print(f"[LeRobotRecorder] Deleting episode {last_episode_index}...")
        
        # Delete parquet file
        parquet_path = self.data_chunk_path / f"episode_{last_episode_index:06d}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            print(f"[LeRobotRecorder] Deleted parquet file: {parquet_path}")
        
        # Delete video files
        if self.config.record_video:
            for cam_name in self.config.camera_names:
                video_path = (self.video_chunk_path / cam_name / 
                             f"episode_{last_episode_index:06d}.mp4")
                if video_path.exists():
                    video_path.unlink()
                    print(f"[LeRobotRecorder] Deleted video: {video_path}")
        
        # Update metadata - remove the last episode
        if self.episodes_info and self.episodes_info[-1]['episode_index'] == last_episode_index:
            deleted_episode = self.episodes_info.pop()
            frames_in_deleted = deleted_episode['length']
            self.total_frames -= frames_in_deleted
        
        if self.episode_stats and self.episode_stats[-1]['episode_index'] == last_episode_index:
            self.episode_stats.pop()
        
        # Update counters
        self.total_episodes -= 1
        self.current_episode_index = self.total_episodes
        self.global_frame_index = self.total_frames
        
        # Save updated metadata
        self.finalize_dataset()
        
        print(f"[LeRobotRecorder] Successfully deleted episode {last_episode_index}")
        print(f"[LeRobotRecorder] Dataset now has {self.total_episodes} episodes, {self.total_frames} frames")
    
    def handle_delete_request(self):
        """Handle delete request with confirmation"""
        current_time = time.time()
        
        if self.total_episodes == 0:
            print("[LeRobotRecorder] No episodes to delete")
            return
        
        if self.is_recording:
            print("[LeRobotRecorder] Cannot delete while recording. Stop recording first.")
            return
        
        if not self.delete_confirmation_pending:
            # First press - set confirmation pending
            self.delete_confirmation_pending = True
            self.delete_confirmation_time = current_time
            last_episode_index = self.total_episodes - 1
            print(f"[LeRobotRecorder] WARNING: Press J again within 3 seconds to confirm deletion of episode {last_episode_index}")
            return
        
        # Check if confirmation is still valid (within 3 seconds)
        if current_time - self.delete_confirmation_time > 3.0:
            # Confirmation expired, treat as first press
            self.delete_confirmation_pending = True
            self.delete_confirmation_time = current_time
            last_episode_index = self.total_episodes - 1
            print(f"[LeRobotRecorder] WARNING: Press J again within 3 seconds to confirm deletion of episode {last_episode_index}")
            return
        
        # Confirmation valid - proceed with deletion
        self.delete_confirmation_pending = False
        print("[LeRobotRecorder] Deletion confirmed!")
        self.delete_previous_recording()
    
    def finalize_dataset(self):
        """Finalize the dataset by writing all metadata files"""
        # Write info.json
        self._write_info_json()
        
        # Write episodes.jsonl
        self._write_episodes_jsonl()
        
        # Write episodes_stats.jsonl (v2.1 format)
        self._write_episodes_stats_jsonl()
        
        # Write tasks.jsonl
        self._write_tasks_jsonl()
        
        print(f"[LeRobotRecorder] Dataset finalized at: {self.dataset_path}")
        print(f"  Episodes: {self.total_episodes}")
        print(f"  Total frames: {self.total_frames}")
    
    def _write_info_json(self):
        """Write the main info.json file"""
        # Define features structure
        features = {
            'observation.state': {
                'dtype': 'float32',
                'shape': [len(self.data.qpos)],
                'names': [f'joint_{i}' for i in range(len(self.data.qpos))]
            },
            'action': {
                'dtype': 'float32', 
                'shape': [7],  # Assuming 7-DOF arm
                'names': [f'action_{i}' for i in range(7)]
            },
            'timestamp': {
                'dtype': 'float64',
                'shape': [],
                'names': None
            },
            'episode_index': {
                'dtype': 'int64',
                'shape': [],
                'names': None
            },
            'frame_index': {
                'dtype': 'int64',
                'shape': [],
                'names': None
            },
            'index': {
                'dtype': 'int64',
                'shape': [],
                'names': None
            },
            'task_index': {
                'dtype': 'int64',
                'shape': [],
                'names': None
            }
        }
        
        # Add video features
        if self.config.record_video:
            for cam_name in self.config.camera_names:
                features[f'observation.images.{cam_name}'] = {
                    'dtype': 'video',
                    'shape': [self.config.video_height, self.config.video_width, 3],
                    'names': ['height', 'width', 'channel'],
                    'info': {
                        'video.fps': self.config.video_fps,
                        'video.codec': 'mp4v',
                        'video.pix_fmt': 'yuv420p',
                        'video.is_depth_map': False,
                        'has_audio': False
                    }
                }
        
        info = {
            'codebase_version': 'v2.1',
            'robot_type': self.config.robot_type,
            'total_episodes': self.total_episodes,
            'total_frames': self.total_frames,
            'total_tasks': 1,
            'total_videos': self.total_episodes * len(self.config.camera_names) if self.config.record_video else 0,
            'fps': self.config.fps,
            'splits': {
                'train': f'0:{self.total_episodes}'
            },
            'features': features
        }
        
        with open(self.meta_path / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def _write_episodes_jsonl(self):
        """Write episodes.jsonl metadata"""
        with open(self.meta_path / 'episodes.jsonl', 'w') as f:
            for episode_info in self.episodes_info:
                f.write(json.dumps(episode_info) + '\n')
    
    def _write_episodes_stats_jsonl(self):
        """Write episodes_stats.jsonl (v2.1 format)"""
        with open(self.meta_path / 'episodes_stats.jsonl', 'w') as f:
            for episode_stats in self.episode_stats:
                f.write(json.dumps(episode_stats) + '\n')
    
    def _write_tasks_jsonl(self):
        """Write tasks.jsonl"""
        with open(self.meta_path / 'tasks.jsonl', 'w') as f:
            task_data = {
                'task_index': 0,
                'task': self.current_task if hasattr(self, 'current_task') else 'robot_manipulation'
            }
            f.write(json.dumps(task_data) + '\n')

    def get_task(self):
        return self.current_task

# Integration functions for control_robot.py
def create_lerobot_recorder(model: mujoco.MjModel, data: mujoco.MjData, 
                           dataset_name: str = "panda_teleop", 
                           output_dir: str = f"{os.getenv('MAIN_DIRECTORY')}/finetuning/datasets",
                           robot_prefix: str = "r1_",
                           use_prefix: bool = True) -> LeRobotDatasetRecorder:
    """Create a LeRobot dataset recorder
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        dataset_name: Name of the dataset
        output_dir: Output directory
        robot_prefix: Prefix for robot (e.g., "r1_", "r2_")
        use_prefix: Whether to use robot prefix for camera names
    """
    config = RecordingConfig(
        dataset_name=dataset_name,
        output_dir=output_dir,
        fps=60,
        video_fps=30,
        use_robot_prefix=use_prefix,
        robot_prefix=robot_prefix,
        video_width=224,
        video_height=224
    )
    return LeRobotDatasetRecorder(model, data, config)

def add_lerobot_controls(recorder: LeRobotDatasetRecorder, on_key_callback):
    """Add LeRobot recording controls to key callback"""
    def enhanced_key_callback(win, key, scancode, action, mods):
        # Call original callback first
        on_key_callback(win, key, scancode, action, mods)
        
        # Add recording controls
        if action == glfw.PRESS:
            if key == glfw.KEY_M:  # M to start/stop recording
                if recorder.is_recording:
                    recorder.stop_recording()
                else:
                    recorder.start_recording()
            elif key == glfw.KEY_N:  # N to finalize dataset
                if recorder.is_recording:
                    recorder.stop_recording()
                recorder.finalize_dataset()
            elif key == glfw.KEY_J:  # J to delete previous recording (requires confirmation)
                recorder.handle_delete_request()
    
    return enhanced_key_callback