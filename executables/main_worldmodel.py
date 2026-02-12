import numpy as np
import os
import math
from omegaconf import OmegaConf
from pathlib import Path as PathLib

from statenav_global.mapping import *
import statenav_global.mapping.WorldModel as WorldModel
from statenav_global.mapping.ros_utils import publish_costmap_float32multiarray, publish_costmap_gridmap

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, Bool
from grid_map_msgs.msg import GridMap

from statenav_global.utility import Utils

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

cfg = OmegaConf.load(PathLib(__file__).parents[0] / "configs/planning_config.yaml")


class WorldModelNode:
    """
    WorldModelNode manages WorldModel, map updates, and optionally global planning.
    
    Architecture:
    - WorldModel runs in THIS process (writer) - updates maps in shared memory
    - Planner runs in SEPARATE process (reader) - reads maps from shared memory
    - Communication: Shared memory (maps) + ROS topics (coordination)
    
    Can run in three modes:
    1. Parent-spawning: shared_metadata provided (multiprocessing)
    2. Named shared memory: use_shared_memory=True, shared_metadata=None
    3. No shared memory: use_shared_memory=False (single process)

    Modes:
    1. Map-only: do_RRT_globalplanning=False (only map updates)
    2. Map + Planning: do_RRT_globalplanning=True (map updates + RRT* planning)
    """

    def __init__(self, shared_metadata=None, init_ros=True, use_shared_memory=None):
        """
        Args:
            shared_metadata: Optional SharedMetadata object from parent process.
                           If provided, uses shared memory with parent-spawning approach.
                           If None, uses config or use_shared_memory parameter.
            init_ros: If True, initialize ROS node. If False, assume ROS is already initialized.
                     Useful when called from multiprocessing workers.
            use_shared_memory: If True, uses shared memory backend. If False, uses local memory.
                              If None, reads from config file or defaults to True if shared_metadata provided.
        """

        os.nice(-10)
        
        # Load and process config
        cfg_path = PathLib(__file__).parent / "configs/planning_config.yaml"
        self.cfg = OmegaConf.load(cfg_path)
        self.rng = Utils.set_random_seed(self.cfg.seed)
        

        # print(OmegaConf.to_yaml(self.cfg))
        print("=" * 80)
        
        # Calculate task parameters
        self.cfg.task_extent = [
            self.cfg.env_extent[0] + self.cfg.local_patch_size,
            self.cfg.env_extent[1] - self.cfg.local_patch_size,
            self.cfg.env_extent[2] + self.cfg.local_patch_size,
            self.cfg.env_extent[3] - self.cfg.local_patch_size
        ]
        self.diagonal = math.hypot(
            self.cfg.task_extent[1] - self.cfg.task_extent[0],
            self.cfg.task_extent[3] - self.cfg.task_extent[2]
        )
        
        self.initial_start = self.cfg.initial_start
        self.global_goal = self.cfg.global_goal
        self.heading_start = np.deg2rad(self.cfg.heading_start)
        self.initialization_time = self.cfg.initialization_time
        self.debugging_visualization = self.cfg.debugging_visualization
        
        # Initialize ROS
        self._init_ros(init_ros)
        self.init_clock = None
        
        # Determine shared memory mode
        self.use_shared_memory = use_shared_memory if use_shared_memory is not None else (
            True if shared_metadata is not None else self.cfg.get('use_shared_memory', False)
        )
        self.shared_metadata = shared_metadata
        
        # Initialize map and world model
        self._init_map_and_worldmodel(shared_metadata)
        
        # Setup ROS communication
        self._setup_ros_communication()
        
        # Initialize planner (if enabled)
        self._init_planner()
        
        rospy.loginfo(f"[WorldModel] WorldModelNode initialized {'with' if self.use_shared_memory else 'without'} shared memory")





####################################### INITIALIZATION #######################################
    def _init_ros(self, init_ros):
        """Initialize ROS node."""
        if init_ros or not rospy.get_node_uri():
            rospy.init_node('WorldModelRunning', anonymous=True)
        rospy.loginfo("[WorldModel] WorldModelNode initialized")

    def _init_map_and_worldmodel(self, shared_metadata):
        """Initialize map and world model based on shared memory mode."""
        if shared_metadata is not None:
            rospy.loginfo("[WorldModel] Initializing WorldModel with shared memory (parent-spawning)")
            self.world_model = WorldModel.WorldModel(use_shared_memory=True, shared_metadata=shared_metadata)
        elif self.use_shared_memory:
            rospy.loginfo("[WorldModel] Initializing WorldModel with shared memory (named)")
            self.world_model = WorldModel.WorldModel(use_shared_memory=True)
        else:
            rospy.loginfo("[WorldModel] Initializing map without shared memory (direct creation)")
            self.world_model = WorldModel.WorldModel(use_shared_memory=False)
            # Map is created by WorldModel.create_map_from_config() in __init__

    def _setup_ros_communication(self):
        """Setup ROS subscribers and publishers."""
        # Subscribers
        rospy.Subscriber("/robot/pose", PoseStamped, self.world_model.global_map.pose_callback, queue_size=1)
        rospy.Subscriber("/elevation_mapping/elevation_map_filter", GridMap,
                        self.world_model.global_map.robo_centric_map_callback, queue_size=1)
        
        # Publishers
        # Replanning signal publisher (for notifying planner when replanning is needed)
        self.replanning_signal_pub = rospy.Publisher("/replanning_signal", Bool, queue_size=1)
        self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
        self.frame_id = self.cfg.frame_id
        
        if not self.use_shared_memory:
            msg_type = GridMap if self.use_gridmap_msg else Float32MultiArray
            self.global_costmap_pub = rospy.Publisher("/global_costmap", msg_type, queue_size=1)
            rospy.loginfo(f"[WorldModel] Using {'GridMap' if self.use_gridmap_msg else 'Float32MultiArray'} for costmap publishing")
        
        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
        self.StaticObs_list_pub = rospy.Publisher("/StaticObs_list", Float32MultiArray, queue_size=1)

    def _init_planner(self):
        """Initialize global planner if enabled."""
        self.do_RRT_globalplanning = self.cfg.do_RRT_globalplanning
        self.asynchronous_globalplanning = self.cfg.asynchronous_globalplanning
        self.pub_locally = self.cfg.pub_locally
        
        # Validate config
        if not self.do_RRT_globalplanning and self.asynchronous_globalplanning:
            raise ValueError("asynchronous_globalplanning requires do_RRT_globalplanning=True")
        if self.asynchronous_globalplanning and self.pub_locally:
            raise ValueError("pub_locally not supported with asynchronous_globalplanning")
        if not self.do_RRT_globalplanning and self.pub_locally:
            raise ValueError("pub_locally requires do_RRT_globalplanning=True")
        
        # Store planner config
        self.localmap_getwaypoint_horizonmultiplier = self.cfg.localmap_getwaypoint_horizonmultiplier
        self.RRT_getwaypoint_steps = self.cfg.RRT_getwaypoint_steps
        self.MPC_horizon = self.cfg.MPC_horizon
        self.step_T = self.cfg.step_T
        
        # Initialize planner
        self.global_planner = None
        if self.do_RRT_globalplanning and not self.asynchronous_globalplanning:
            self.global_planner = statenav_global.planners.GlobalRRTStar(
                self.cfg.task_extent, self.rng, self.initial_start, self.global_goal, self.heading_start,
                goal_radius=self.diagonal * self.cfg.goal_radius_ratio,
                branch_length_max=self.cfg.branch_length_max_ratio * self.diagonal,
                search_radius=self.cfg.search_radius_ratio * self.diagonal,
                decrease_search_radius=True,
                iter_max=self.cfg.iter_max,
                convergence_threshold=self.cfg.convergence_ratio,
                switch_to_informed_from_thisiter=self.cfg.switch_to_informed_from_thisiter,
                sampling_dist=self.cfg.sampling_dist,
                num_samplingpoints=self.cfg.num_samplingpoints,
                default_obstacle_clearance=self.cfg.obs_clearance,
                movable_obs_clearance=0.2,
                max_clearout_effort=1.5,
                robot_radius=0.3
            )
            self.global_planner.global_map = self.world_model.global_map
            rospy.loginfo("[WorldModel] Initialized planner GlobalRRTStar")
        else:
            rospy.loginfo("[WorldModel] Global planning disabled")









####################################### UPDATE MAP #######################################
    def _update_map(self):
        """Update the global map."""
        path_plan = self.global_planner.path if (self.do_RRT_globalplanning and 
                                                 not self.asynchronous_globalplanning and 
                                                 self.global_planner) else None
        
        self.world_model.global_map.Update_map(
            path_plan=path_plan,
            visualize_map=self.debugging_visualization
        )

####################################### RUN PLANNER #######################################
    def _run_planner(self):
        """Run planner replanning if enabled."""
        
        self.global_planner.replan(
            initial_start=(self.initial_start[0], self.initial_start[1]),
            step_T=self.step_T,
            RRT_getwaypoint_steps=self.RRT_getwaypoint_steps,
            plot_map=True
        )
        
        # Debug output
        rospy.loginfo(f"[WorldModel] Planner cost: {self.global_planner.cost(self.global_planner.s_goal)}")

####################################### PUBLISH  #######################################
    def _publish_costmap(self):
        """Publish costmap."""
        publish_func = publish_costmap_gridmap if self.use_gridmap_msg else publish_costmap_float32multiarray
        publish_func(
            self.world_model.global_map, self.frame_id, self.global_costmap_pub,
            self.pub_locally, self.global_planner,
            self.step_T, self.localmap_getwaypoint_horizonmultiplier, self.MPC_horizon
        )

    def _publish_StaticObs_list(self):
        """Publish obstacle list."""
        msg = Float32MultiArray()
        for obs in self.world_model.global_map.StaticObs_list:
            msg.data.extend(obs[:2])
        self.StaticObs_list_pub.publish(msg)

    def _publish_path(self):
        """Publish global path if planner is enabled."""
        if not (self.do_RRT_globalplanning and not self.asynchronous_globalplanning and self.global_planner):
            return
        
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = rospy.Time.now()
        
        for pt in self.global_planner.path:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pt[0], pt[1], 1.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.global_path_pub.publish(path_msg)

    def _handle_replanning_signals(self):
        """Handle replanning signals via ROS topic."""
        if self.world_model.send_replanning:
            self.replanning_signal_pub.publish(Bool(True))
            self.world_model.send_replanning = False











####################################### RUN #######################################
    def run(self):
        """Main execution loop."""
        rate = rospy.Rate(1)
        rospy.loginfo("[WorldModel] WorldModelNode running. Waiting for map data...")
        
        count = 0
        while not rospy.is_shutdown():
            count += 1
            rospy.loginfo_throttle(1.0, "[WorldModel] \n==================================== LOOP ====================================")
            rospy.loginfo_throttle(1.0, "[WorldModel] ROSPY in loop. ROSTime is, %f and count is %d", rospy.get_time(), count)
            
            try:
                if rospy.get_time() > self.initialization_time:
                    if self.init_clock is None:
                        self.init_clock = rospy.get_time()
                    






                    
                    self._update_map()
                    
                    if self.world_model.global_map.is_TraversabilityMap_built:

                        if not self.use_shared_memory:
                            self._publish_costmap()
                        self._publish_StaticObs_list()

                        if (self.do_RRT_globalplanning and not self.asynchronous_globalplanning and self.global_planner):
                            self._run_planner()
                            self._publish_path()

                    self.world_model.send_replanning = True
                    self._handle_replanning_signals()
                
            except Exception as e:
                rospy.logerr(f"[WorldModel] Error in WorldModelNode run loop: {e}")
                import traceback
                rospy.logerr(f"[WorldModel] {traceback.format_exc()}")
            
            rate.sleep()



####################################### MAIN #######################################
def main():
    try:
        node = WorldModelNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.loginfo("[WorldModel] WorldModelNode shutting down")
        rospy.signal_shutdown("Finished execution")


if __name__ == '__main__':
    main()
