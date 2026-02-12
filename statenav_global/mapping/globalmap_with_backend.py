"""
Modified CMDbasedMap that uses SharedMemoryBackend

This shows how to modify CMDbasedMap to use the backend pattern.
The key insight: CMDbasedMap gets arrays from backend instead of creating them itself.
All methods work unchanged - they just use arrays from backend.

IMPORTANT: Call backend.cleanup() on shutdown to free shared memory!
"""

import numpy as np
import atexit
try:
    import rospy
except ImportError:
    rospy = None

from statenav_global.mapping.globalmap import CMDbasedMap, BaseMap
from statenav_global.mapping.shared_memory_backend import SharedMemoryBackend, LocalMemoryBackend


class CMDbasedMapWithBackend(CMDbasedMap):
    """
    CMDbasedMap that uses SharedMemoryBackend
    
    Key changes:
    1. Accepts backend in __init__
    2. Gets arrays from backend instead of creating them
    3. Uses backend for synchronization
    4. All existing methods work unchanged!
    """
    
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y,
                 which_layer, preest_update_resolution, instab_limit,
                 load_Travformer=True, load_MapReconstructor=False,
                 backend=None, use_shared_memory=False):
        """
        Initialize CMDbasedMap with backend
        
        Args:
            backend: SharedMemoryBackend or LocalMemoryBackend instance
                     If None, creates appropriate backend based on use_shared_memory
            use_shared_memory: If True and backend=None, creates SharedMemoryBackend
        """
        # Initialize parent class (but don't create arrays yet)
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y,
                        which_layer, preest_update_resolution, instab_limit,
                        load_Travformer, load_MapReconstructor)
        
        # Set up backend
        if backend is None:
            if use_shared_memory:
                self.backend = SharedMemoryBackend(mode='writer')
            else:
                self.backend = LocalMemoryBackend()
        else:
            self.backend = backend
        
        self.use_shared_memory = use_shared_memory or isinstance(self.backend, SharedMemoryBackend)
        
        # Register cleanup handlers if using shared memory
        if self.use_shared_memory:
            atexit.register(self._cleanup_handler)
            if rospy is not None:
                rospy.on_shutdown(self._cleanup_handler)
        
        # Arrays will be set from backend in Initilize_map()
        # We'll override Initilize_map to use backend
    
    def _cleanup_handler(self):
        """Cleanup handler for atexit/ROS shutdown"""
        if hasattr(self, 'backend') and self.use_shared_memory:
            try:
                self.backend.cleanup()
            except Exception:
                pass  # Ignore errors during shutdown
    
    def Initilize_map(self):
        """
        Initialize maps using backend
        
        This replaces the original Initilize_map to use backend arrays.
        Note: If using shared memory, arrays are already initialized in robo_centric_map_callback.
        """
        if self.is_RobocentricMap_received:
            if not self.is_ElevationMap_built:
                self.is_ElevationMap_built = True
            
            # Calculate dimensions
            self.ElevationMap_size_rows = int(self.ElevationMap_size_x / self.map_resolution)
            self.ElevationMap_size_cols = int(self.ElevationMap_size_y / self.map_resolution)
            
            self.TraversabilityMap_size_rows = int(self.TraversabilityMap_size_x / self.map_resolution)
            self.TraversabilityMap_size_cols = int(self.TraversabilityMap_size_y / self.map_resolution)
            
            self.ElevMap_MetaInfo_size_rows = int(self.ElevMap_MetaInfo_size_x / self.map_resolution)
            self.ElevMap_MetaInfo_size_cols = int(self.ElevMap_MetaInfo_size_y / self.map_resolution)
            
            self.TravMap_MetaInfo_size_rows = int(self.TravMap_MetaInfo_size_x / self.map_resolution)
            self.TravMap_MetaInfo_size_cols = int(self.TravMap_MetaInfo_size_y / self.map_resolution)
            
            # Initialize backend with dimensions
            self.backend.initialize_arrays(
                elev_map_rows=self.ElevationMap_size_rows,
                elev_map_cols=self.ElevationMap_size_cols,
                trav_map_rows=self.TraversabilityMap_size_rows,
                trav_map_cols=self.TraversabilityMap_size_cols,
                trav_map_layers=self.TraversabilityMap_size_layers,
                map_resolution=self.map_resolution,
                elev_map_xmin=self.ElevationMap_xmin,
                elev_map_xmax=self.ElevationMap_xmax,
                elev_map_ymin=self.ElevationMap_ymin,
                elev_map_ymax=self.ElevationMap_ymax
            )
            
            # Get arrays from backend (these point to shared memory if using it)
            self.TraversabilityMap = self.backend.get_traversability_map()
            self.ElevationMap = self.backend.get_elevation_map()
            self.ElevMap_MetaInfo = self.backend.get_elev_meta_info()
            self.TravMap_MetaInfo = self.backend.get_trav_meta_info()
            
            # Initialize with default values (same as original)
            self.TraversabilityMap[:,:,:,0] = self.InitialGuess_cmd_v
            self.TraversabilityMap[:,:,:,1] = self.InitialGuess_cmd_w
            self.TraversabilityMap[:,:,:,2] = self.InitialGuess_auxiliary_score
            self.TraversabilityMap[:,:,:,3] = self.InitialGuess_auxiliary_score_std
            
            # Ensure TraversabilityMap bounds are set (needed for xy2grid, etc.)
            self.TraversabilityMap_xmin = self.ElevationMap_xmin
            self.TraversabilityMap_xmax = self.ElevationMap_xmax
            self.TraversabilityMap_ymin = self.ElevationMap_ymin
            self.TraversabilityMap_ymax = self.ElevationMap_ymax
            
            # Mark as built in backend
            self.backend.set_elev_map_built(True)
            
            if not self.is_TraversabilityMap_built:
                self.is_TraversabilityMap_built = True
                self.backend.set_trav_map_built(True)
    
    def Update_map(self, path_plan=None, visualize_map=False):
        """
        Update map with synchronization
        
        This wraps the original Update_map with locking
        """
        if self.use_shared_memory:
            # Acquire write lock before updating
            self.backend.acquire_write_lock()
        
        try:
            # Call parent's Update_map (all existing code works!)
            super().Update_map(path_plan, visualize_map)
            
            # After updates, increment version and update flags
            if self.use_shared_memory:
                self.backend.set_trav_map_built(self.is_TraversabilityMap_built)
                self.backend.set_elev_map_built(self.is_ElevationMap_built)
                self.backend.increment_version()
        
        finally:
            if self.use_shared_memory:
                # Always release lock
                self.backend.release_write_lock()
    
    def pose_callback(self, msg):
        """
        Update robot pose (also update in shared memory)
        """
        # Call parent's pose_callback
        super().pose_callback(msg)
        
        # Also update in shared memory backend
        if self.use_shared_memory:
            self.backend.set_robot_pose(self.robot_x, self.robot_y, self.robot_heading)
    
    def get_backend(self):
        """Get backend instance (for access to metadata, version, etc.)"""
        return self.backend
    
    def cleanup(self):
        """Explicit cleanup (also called automatically on shutdown)"""
        if self.use_shared_memory:
            self.backend.cleanup()


class CMDbasedMapReaderProxy(CMDbasedMap):
    """
    Proxy class for Planner to access CMDbasedMap via shared memory
    
    This inherits from CMDbasedMap so all methods work unchanged.
    The only difference: arrays come from shared memory backend.
    """
    
    def __init__(self, backend: SharedMemoryBackend):
        """
        Initialize proxy with shared memory backend
        
        Args:
            backend: SharedMemoryBackend instance (mode='reader')
        """
        self.backend = backend
        
        # Wait for writer to initialize metadata and shared memory
        # Then initialize backend arrays
        self._wait_and_initialize_backend()
        
        # Get dimensions and bounds from backend (now initialized)
        dims = backend.get_map_dimensions()
        bounds = backend.get_map_bounds()
        
        # Initialize parent CMDbasedMap with minimal parameters
        # (We'll override the arrays anyway)
        super().__init__(
            env_xmin=bounds[0],
            env_xmax=bounds[1],
            env_ymin=bounds[2],
            env_ymax=bounds[3],
            goal_x=0.0,  # Will be set separately if needed
            goal_y=0.0,
            which_layer=[],  # Not used in reader
            preest_update_resolution=0.0,  # Not used in reader
            instab_limit=0.0,  # Not used in reader
            load_Travformer=False,  # Don't load models in reader
            load_MapReconstructor=False
        )
        
        # Register cleanup handlers
        atexit.register(self._cleanup_handler)
        if rospy is not None:
            rospy.on_shutdown(self._cleanup_handler)
        
        # Override arrays with shared memory arrays
        self.TraversabilityMap = backend.get_traversability_map()
        self.ElevationMap = backend.get_elevation_map()
        self.ElevMap_MetaInfo = backend.get_elev_meta_info()
        self.TravMap_MetaInfo = backend.get_trav_meta_info()
        
        # Set map parameters from backend metadata
        self.map_resolution = backend.get_map_resolution()
        bounds = backend.get_map_bounds()
        self.ElevationMap_xmin = bounds[0]
        self.ElevationMap_xmax = bounds[1]
        self.ElevationMap_ymin = bounds[2]
        self.ElevationMap_ymax = bounds[3]
        
        # TraversabilityMap uses same bounds as ElevationMap
        self.TraversabilityMap_xmin = bounds[0]
        self.TraversabilityMap_xmax = bounds[1]
        self.TraversabilityMap_ymin = bounds[2]
        self.TraversabilityMap_ymax = bounds[3]
        
        # Set dimensions
        dims = backend.get_map_dimensions()
        self.TraversabilityMap_size_rows = dims['trav_map_rows']
        self.TraversabilityMap_size_cols = dims['trav_map_cols']
        self.TraversabilityMap_size_layers = dims['trav_map_layers']
        self.ElevationMap_size_rows = dims['elev_map_rows']
        self.ElevationMap_size_cols = dims['elev_map_cols']
        
        # Size attributes for consistency (calculated from bounds)
        self.ElevationMap_size_x = self.ElevationMap_xmax - self.ElevationMap_xmin
        self.ElevationMap_size_y = self.ElevationMap_ymax - self.ElevationMap_ymin
        self.TraversabilityMap_size_x = self.TraversabilityMap_xmax - self.TraversabilityMap_xmin
        self.TraversabilityMap_size_y = self.TraversabilityMap_ymax - self.TraversabilityMap_ymin
        
        # MetaInfo size attributes (same as their corresponding maps)
        self.ElevMap_MetaInfo_xmin = self.ElevationMap_xmin
        self.ElevMap_MetaInfo_xmax = self.ElevationMap_xmax
        self.ElevMap_MetaInfo_ymin = self.ElevationMap_ymin
        self.ElevMap_MetaInfo_ymax = self.ElevationMap_ymax
        self.ElevMap_MetaInfo_size_x = self.ElevationMap_size_x
        self.ElevMap_MetaInfo_size_y = self.ElevationMap_size_y
        self.ElevMap_MetaInfo_size_rows = dims['elev_map_rows']
        self.ElevMap_MetaInfo_size_cols = dims['elev_map_cols']
        
        self.TravMap_MetaInfo_xmin = self.TraversabilityMap_xmin
        self.TravMap_MetaInfo_xmax = self.TraversabilityMap_xmax
        self.TravMap_MetaInfo_ymin = self.TraversabilityMap_ymin
        self.TravMap_MetaInfo_ymax = self.TraversabilityMap_ymax
        self.TravMap_MetaInfo_size_x = self.TraversabilityMap_size_x
        self.TravMap_MetaInfo_size_y = self.TraversabilityMap_size_y
        self.TravMap_MetaInfo_size_rows = dims['trav_map_rows']
        self.TravMap_MetaInfo_size_cols = dims['trav_map_cols']
        self.TravMap_MetaInfo_size_layers = dims['trav_map_layers']
        
        # Map parameters (same as CMDbasedMap)
        self.TraversabilityMap_theta_min = -np.pi
        self.TraversabilityMap_theta_resolution = np.pi / 4
        self.TraversabilityMap_theta_max = np.pi - self.TraversabilityMap_theta_resolution
        
        # TravMap_MetaInfo theta parameters
        self.TravMap_MetaInfo_theta_min = self.TraversabilityMap_theta_min
        self.TravMap_MetaInfo_theta_resolution = self.TraversabilityMap_theta_resolution
        self.TravMap_MetaInfo_theta_max = self.TraversabilityMap_theta_max
        
        # State flags (initial values, will be updated from backend)
        self.is_TraversabilityMap_built = backend.get_trav_map_built()
        self.is_ElevationMap_built = backend.get_elev_map_built()
        self.is_ElevMap_MetaInfo_built = True  # MetaInfo arrays are available from shared memory
        self.is_TravMap_MetaInfo_built = True
    
    def update_map_flags(self):
        """Update map built flags from shared memory backend."""
        if hasattr(self, 'backend'):
            self.is_TraversabilityMap_built = self.backend.get_trav_map_built()
            self.is_ElevationMap_built = self.backend.get_elev_map_built()
    
    def _wait_and_initialize_backend(self, timeout: float = 30.0, check_interval: float = 0.1):
        """
        Wait for writer to initialize metadata AND create shared memory blocks, then initialize backend arrays.
        
        Args:
            timeout: Maximum time to wait for writer initialization (seconds)
            check_interval: Time between checks (seconds)
        """
        import time
        from multiprocessing import shared_memory
        
        start_time = time.time()
        if rospy is not None:
            rospy.loginfo("[PathPlanner] Waiting for WorldModel to initialize shared memory...")
        
        while time.time() - start_time < timeout:
            # Check if metadata has been initialized (dimensions > 0)
            dims = self.backend.get_map_dimensions()
            if dims['trav_map_rows'] > 0 and dims['elev_map_rows'] > 0:
                # Metadata is ready, check if shared memory blocks exist
                # Try to attach to all required shared memory blocks to verify they exist
                
                # TODO: Why?
                try:
                    test_shm_trav = shared_memory.SharedMemory(name='trav_map_shm')
                    test_shm_elev = shared_memory.SharedMemory(name='elev_map_shm')
                    test_shm_elev_meta = shared_memory.SharedMemory(name='elev_meta_shm')
                    test_shm_trav_meta = shared_memory.SharedMemory(name='trav_meta_shm')
                    # Close test connections
                    test_shm_trav.close()
                    test_shm_elev.close()
                    test_shm_elev_meta.close()
                    test_shm_trav_meta.close()
                    
                    # All shared memory blocks exist, now initialize arrays (this will attach properly)
                    bounds = self.backend.get_map_bounds()
                    self.backend.initialize_arrays(
                        elev_map_rows=dims['elev_map_rows'],
                        elev_map_cols=dims['elev_map_cols'],
                        trav_map_rows=dims['trav_map_rows'],
                        trav_map_cols=dims['trav_map_cols'],
                        trav_map_layers=dims['trav_map_layers'],
                        map_resolution=self.backend.get_map_resolution(),
                        elev_map_xmin=bounds[0],
                        elev_map_xmax=bounds[1],
                        elev_map_ymin=bounds[2],
                        elev_map_ymax=bounds[3]
                    )
                    if rospy is not None:
                        rospy.loginfo("[Reader] Successfully attached to shared memory")
                    return
                except FileNotFoundError:
                    # Shared memory blocks don't exist yet, keep waiting
                    # This can happen if metadata is set before blocks are created
                    pass
                except Exception as e:
                    # Other errors - log and continue waiting
                    if rospy is not None:
                        rospy.logwarn_throttle(5.0, f"[Reader] Error checking shared memory: {e}")
            
            time.sleep(check_interval)
            if rospy is not None:
                rospy.loginfo_throttle(2.0, "[Reader] Still waiting for WorldModel to initialize shared memory...")
        
        raise RuntimeError(f"Timeout waiting for WorldModel to initialize shared memory ({timeout}s)")
    
    def update_robot_pose(self):
        """Update robot pose from shared memory"""
        robot_pose = self.backend.get_robot_pose()
        self.robot_x = robot_pose[0]
        self.robot_y = robot_pose[1]
        self.robot_heading = robot_pose[2]
    
    def check_version(self):
        """Check if maps have been updated"""
        return self.backend.get_version()
    
    def acquire_read_lock(self):
        """
        Acquire read lock (OPTIONAL - for consistency guarantees)
        
        Read lock is SHARED - multiple readers can hold it.
        
        Use when:
        - Reading multiple elements that must be consistent together
        - Want to prevent reading during bulk updates
        
        Don't need for:
        - Individual element reads (atomic): trav_map[row, col, layer, 0]
        - Most planning operations (they read individual elements)
        """
        self.backend.acquire_read_lock()
    
    def release_read_lock(self):
        """Release read lock"""
        self.backend.release_read_lock()
    
    def _cleanup_handler(self):
        """Cleanup handler for atexit/ROS shutdown"""
        if hasattr(self, 'backend'):
            try:
                self.backend.cleanup()
            except Exception:
                pass  # Ignore errors during shutdown
    
    def cleanup(self):
        """Explicit cleanup (also called automatically on shutdown)"""
        self.backend.cleanup()

