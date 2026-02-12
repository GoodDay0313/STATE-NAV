import numpy as np
try:
    import rospy
    from std_msgs.msg import Bool
except ImportError:
    rospy = None

from statenav_global.mapping.globalmap import CMDbasedMap, LearnedInSMap
from statenav_global.mapping.globalmap_with_backend import CMDbasedMapWithBackend
from statenav_global.utility.utils import get_project_root
from pathlib import Path
from omegaconf import OmegaConf

# Load config (relative to project root)
cfg = OmegaConf.load(get_project_root() / "executables/configs/planning_config.yaml")


class WorldModel:
    """
    WorldModel manages the global map and obstacles.
    
    When use_shared_memory=True:
    - Uses CMDbasedMapWithBackend (writer mode)
    - Creates shared memory that Planner (separate process) can read
    - Maps are updated in shared memory, accessible by Planner
    """
    
    @staticmethod
    def create_map_from_config(cfg=None, global_goal=None):
        """
        Create map instance based on trav_option and planner_option configs.
        
        Args:
            cfg: Optional config object. If None, uses module-level cfg.
            global_goal: Optional global goal [x, y]. If None, uses cfg.global_goal.
        
        Returns:
            Map instance (CMDbasedMap, LearnedInSMap, IHMCMap, or QuadrupedMap)
        """
        if cfg is None:
            cfg = globals()['cfg']
        if global_goal is None:
            global_goal = cfg.global_goal
        
        map_params = {
            'env_xmin': cfg.env_extent[0], 'env_xmax': cfg.env_extent[1],
            'env_ymin': cfg.env_extent[2], 'env_ymax': cfg.env_extent[3],
            'goal_x': global_goal[0], 'goal_y': global_goal[1],
            'which_layer': cfg.which_layer,
            'preest_update_resolution': cfg.trav_estimation_resoultion,
            'instab_limit': cfg.instability_limit
        }
        
        if cfg.trav_option == "Proposed" and cfg.planner_option == "safecmd":
            return CMDbasedMap(load_Travformer=True, load_MapReconstructor=False, **map_params)
        elif cfg.trav_option == "Proposed" and cfg.planner_option == "score":
            return LearnedInSMap(**map_params)
        else:
            raise ValueError(f"Invalid trav_option: {cfg.trav_option}")

    def __init__(self, use_shared_memory=False, backend=None, shared_metadata=None):
        """
        Initialize WorldModel
        
        Args:
            use_shared_memory: If True, uses shared memory backend for maps
                              (enables multiprocess access by Planner)
            backend: Optional SharedMemoryBackend or LocalMemoryBackend instance.
                    If provided, uses this backend instead of creating a new one.
            shared_metadata: Optional SharedMetadata object from parent process.
                           If provided, creates SharedMemoryBackend with this metadata.
                           Only used if backend=None and use_shared_memory=True.
        """
        
        # Initialize map with or without shared memory backend
        if use_shared_memory or backend is not None:
            # Use shared memory backend (writer mode)
            # This creates shared memory that Planner can attach to
            if backend is None:
                # Create backend with shared_metadata if provided
                from statenav_global.mapping.shared_memory_backend import SharedMemoryBackend
                if shared_metadata is not None:
                    backend = SharedMemoryBackend(mode='writer', shared_metadata=shared_metadata)
                else:
                    backend = SharedMemoryBackend(mode='writer')
            
            self.global_map = CMDbasedMapWithBackend(
                env_xmin=cfg.env_extent[0],
                env_xmax=cfg.env_extent[1],
                env_ymin=cfg.env_extent[2],
                env_ymax=cfg.env_extent[3],
                goal_x=cfg.global_goal[0],
                goal_y=cfg.global_goal[1],
                which_layer=cfg.which_layer,
                preest_update_resolution=cfg.trav_estimation_resoultion,
                instab_limit=cfg.instability_limit,
                load_Travformer=True,
                load_MapReconstructor=False,
                backend=backend,  # Pass the backend directly
                use_shared_memory=True
            )
            if rospy:
                if shared_metadata is not None:
                    rospy.loginfo("[WorldModel] WorldModel initialized with shared memory backend (writer mode, parent-spawning)")
                else:
                    rospy.loginfo("[WorldModel] WorldModel initialized with shared memory backend (writer mode)")
        else:
            # Use regular map (single process) - create based on config
            self.global_map = WorldModel.create_map_from_config()
        
        self.RGB_view = None
        self.depth_view = None
        self.send_replanning = False
        












    