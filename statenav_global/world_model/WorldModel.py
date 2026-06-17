try:
    import rospy
except ImportError:
    rospy = None

from statenav_global.world_model.globalmap import CMDbasedMap
from statenav_global.world_model.globalmap_with_backend import CMDbasedMapWithBackend
from pathlib import Path
from omegaconf import OmegaConf

# Load config from package data
_PKG_ROOT = Path(__file__).resolve().parent.parent
cfg = OmegaConf.load(_PKG_ROOT / "configs/planning_config.yaml")


class WorldModel:
    """
    WorldModel manages the global elevation and traversability maps.

    When use_shared_memory=True:
    - Uses CMDbasedMapWithBackend (writer mode)
    - Creates shared memory that Planner (separate process) can read
    - Maps are updated in shared memory, accessible by Planner
    """

    @staticmethod
    def create_map_from_config(cfg=None, global_goal=None):
        """
        Create map instance based on trav_option config.

        Args:
            cfg: Optional config object. If None, uses module-level cfg.
            global_goal: Optional global goal [x, y]. If None, uses cfg.global_goal.

        Returns:
            Map instance (CMDbasedMap)
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

        if cfg.trav_option == "Proposed":
            return CMDbasedMap(load_Travformer=True, **map_params)
        else:
            raise ValueError(f"Invalid trav_option: {cfg.trav_option}")

    def __init__(self, use_shared_memory=False, backend=None):
        """
        Initialize WorldModel

        Args:
            use_shared_memory: If True, uses shared memory backend for maps
                              (enables multiprocess access by Planner)
            backend: Optional SharedMemoryBackend or LocalMemoryBackend instance.
                    If provided, uses this backend instead of creating a new one.
        """
        if use_shared_memory or backend is not None:
            if backend is None:
                from statenav_global.world_model.shared_memory_backend import SharedMemoryBackend
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
                backend=backend,
                use_shared_memory=True
            )
            if rospy:
                rospy.loginfo("[WorldModel] WorldModel initialized with shared memory backend (writer mode)")
        else:
            self.global_map = WorldModel.create_map_from_config()
