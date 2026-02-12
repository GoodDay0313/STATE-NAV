from time import time
import numpy as np
import math
from omegaconf import OmegaConf

from statenav_global.planners import GlobalRRTStar
from statenav_global.mapping import CMDbasedMap, LearnedInSMap
from statenav_global.mapping.globalmap_with_backend import CMDbasedMapReaderProxy
from statenav_global.mapping.shared_memory_backend import SharedMemoryBackend
from statenav_global.mapping.ros_utils import populate_map_from_float32multiarray, populate_map_from_gridmap



import transforms3d as tf3
import tf.transformations as tf_trans





import rospy
from std_msgs.msg import Float32MultiArray, Bool
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pathlib import Path as PathLib
cfg = OmegaConf.load(PathLib(__file__).parents[0] / "configs/planning_config.yaml")


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    print(f"Set random seed to {seed} in numpy.")
    return rng


def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))






















################################################### Helper Functions for ROS ###################################################





































class PlanningNode:
    """Main planning node that subscribes to map and publishes paths."""
    
    def __init__(self, shared_metadata=None, init_ros=True):
        """
        Initialize PlanningNode
        
        Args:
            shared_metadata: Optional SharedMetadata object from parent process.
                           If provided, uses shared memory with parent-spawning approach.
                           If None, uses config to determine mode (named shared memory or ROS messages).
            init_ros: If True, initialize ROS node. If False, assume ROS is already initialized.
                     Useful when called from multiprocessing workers.
        """
        # Load configuration
        self.cfg = cfg
        
        # Check if using shared memory mode
        self.use_shared_memory = self.cfg.get('use_shared_memory', False) or (shared_metadata is not None)
        self.shared_metadata = shared_metadata  # Store for parent-spawning approach
        
        # Calculate task extent
        self.task_extent = [
            self.cfg.env_extent[0] + 1*self.cfg.local_patch_size,
            self.cfg.env_extent[1] - 1*self.cfg.local_patch_size,
            self.cfg.env_extent[2] + 1*self.cfg.local_patch_size,
            self.cfg.env_extent[3] - 1*self.cfg.local_patch_size
        ]
        diagonal = math.hypot(
            self.task_extent[1] - self.task_extent[0],
            self.task_extent[3] - self.task_extent[2]
        )
        
        global_goal = self.cfg.global_goal
        
        # Initialize map based on mode
        if self.use_shared_memory:
            # Shared memory mode: attach to shared memory created by WorldModel
            if shared_metadata is not None:
                rospy.loginfo("[PathPlanner] Initializing in SHARED MEMORY mode (reader, parent-spawning)")
                backend = SharedMemoryBackend(mode='reader', shared_metadata=shared_metadata)
            else:
                rospy.loginfo("[PathPlanner] Initializing in SHARED MEMORY mode (reader, named)")
                backend = SharedMemoryBackend(mode='reader')
            
            # Note: CMDbasedMapReaderProxy only supports CMDbasedMap for now
            # If you need other map types, you'll need to create reader proxies for them
            if self.cfg.trav_option == "Proposed" and self.cfg.planner_option == "safecmd":
                self.global_map = CMDbasedMapReaderProxy(backend)
                # Set goal after initialization
                self.global_map.goal_x = global_goal[0]
                self.global_map.goal_y = global_goal[1]
            else:
                rospy.logwarn(f"[PathPlanner] Shared memory mode currently only supports CMDbasedMap with safecmd option.")
                rospy.logwarn(f"[PathPlanner] Requested: {self.cfg.trav_option}/{self.cfg.planner_option}. Falling back to ROS message mode.")
                self.use_shared_memory = False
                self.shared_metadata = None
        
        if not self.use_shared_memory:
            # ROS message mode: create map and subscribe to topics
            rospy.loginfo("[PathPlanner] Initializing in ROS MESSAGE mode")
            if self.cfg.trav_option == "Proposed" and self.cfg.planner_option == "safecmd":
                self.global_map = CMDbasedMap(
                    env_xmin=self.cfg.env_extent[0], 
                    env_xmax=self.cfg.env_extent[1], 
                    env_ymin=self.cfg.env_extent[2], 
                    env_ymax=self.cfg.env_extent[3],
                    goal_x=global_goal[0], 
                    goal_y=global_goal[1],
                    which_layer=self.cfg.which_layer,
                    preest_update_resolution=self.cfg.trav_estimation_resoultion, 
                    instab_limit=self.cfg.instability_limit,
                    load_Travformer = False
                )
            elif self.cfg.trav_option == "Proposed" and self.cfg.planner_option == "score":
                self.global_map = LearnedInSMap(
                    env_xmin=self.cfg.env_extent[0], 
                    env_xmax=self.cfg.env_extent[1], 
                    env_ymin=self.cfg.env_extent[2], 
                    env_ymax=self.cfg.env_extent[3],
                    goal_x=global_goal[0], 
                    goal_y=global_goal[1],
                    which_layer=self.cfg.which_layer,
                    preest_update_resolution=self.cfg.trav_estimation_resoultion, 
                    instab_limit=self.cfg.instability_limit,
                    load_Travformer = False
                )
            else:
                raise ValueError(f"Invalid trav_option or planner_option: {self.cfg.trav_option} or {self.cfg.planner_option}")
        
        rospy.loginfo(f"[PathPlanner] Map class initialized. Type: {type(self.global_map).__name__}, Mode: {'SHARED MEMORY' if self.use_shared_memory else 'ROS MESSAGE'}")
        
        # Initialize RRT planner
        self.initial_start = self.cfg.initial_start
        self.global_goal = self.cfg.global_goal
        self.heading_start = np.deg2rad(self.cfg.heading_start)
        
        self.rng = set_random_seed(self.cfg.seed)
        
        iter_max_global = self.cfg.iter_max
        branch_length_max = self.cfg.branch_length_max_ratio * diagonal
        search_radius = self.cfg.search_radius_ratio * diagonal
        
        self.global_planner = GlobalRRTStar(
            self.task_extent, self.rng, self.initial_start, self.global_goal, self.heading_start,
            goal_radius=diagonal * self.cfg.goal_radius_ratio,
            branch_length_max=branch_length_max,
            search_radius=search_radius,
            decrease_search_radius=True,
            iter_max=iter_max_global,
            convergence_threshold=self.cfg.convergence_ratio,
            switch_to_informed_from_thisiter=self.cfg.switch_to_informed_from_thisiter,
            sampling_dist=self.cfg.sampling_dist,
            num_samplingpoints=self.cfg.num_samplingpoints,
            default_obstacle_clearance=self.cfg.obs_clearance
        )
        self.global_planner.global_map = self.global_map
        
        # ROS setup (conditional initialization)
        if init_ros:
            rospy.init_node('Planning_Node', anonymous=True)
        elif not rospy.get_node_uri():
            # ROS not initialized and init_ros=False - this shouldn't happen
            rospy.logwarn("[PathPlanner] ROS not initialized but init_ros=False. Initializing anyway...")
            rospy.init_node('Planning_Node', anonymous=True)
        
        self.frame_id = self.cfg.get('frame_id', 'map')
        
        # Subscribers (different based on mode)
        if self.use_shared_memory:
            # Shared memory mode: subscribe to pose and replanning signal via ROS
            rospy.Subscriber("/robot/pose", PoseStamped, self.pose_callback, queue_size=1)
            rospy.Subscriber("/replanning_signal", Bool, self.replanning_callback, queue_size=1)
            rospy.loginfo("[PathPlanner] Subscribed to /robot/pose and /replanning_signal (shared memory mode)")
        else:
            # ROS message mode: subscribe to map topics and pose
            self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
            if self.use_gridmap_msg:
                rospy.Subscriber("/global_costmap", GridMap, self.map_callback, queue_size=1)
                rospy.loginfo("[PathPlanner] Subscribed to /global_costmap (GridMap)")
            else:
                rospy.Subscriber("/global_costmap", Float32MultiArray, self.map_callback, queue_size=1)
                rospy.loginfo("[PathPlanner] Subscribed to /global_costmap (Float32MultiArray)")
            
            rospy.Subscriber("/robot/pose", PoseStamped, self.pose_callback, queue_size=1)
            rospy.loginfo("[PathPlanner] Subscribed to /robot/pose")
        
        print(self.global_map.robot_heading)
        
        # Publishers
        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
        rospy.loginfo("[PathPlanner] Publisher initialized: /global_path")
        
        # For getting next waypoint for resetting the tree
        self.step_T = self.cfg.step_T
        self.RRT_getwaypoint_steps = self.cfg.RRT_getwaypoint_steps
        
        # State
        self.last_plan_time = 0
        self.planning_in_progress = False
        self.initialization_time = self.cfg.initialization_time
        self.replanning_needed = True  # For shared memory mode
        # TODO: Handle replanning needed later
        
        rospy.loginfo(f"[PathPlanner] Planning Node initialized with {self.cfg.trav_option}/{self.cfg.planner_option} map type!")
    





    def plan_and_publish(self):
        """Run RRT planning and publish the path."""
        if not self.global_map.is_TraversabilityMap_built:
            rospy.logwarn("[PathPlanner] Map not built yet, cannot plan")
            return
        
        self.planning_in_progress = True
        rospy.loginfo("[PathPlanner] Planning in progress")
        
        try:
            if self.use_shared_memory:
                # Update robot pose from shared memory before planning
                self.global_map.update_robot_pose()
                # Use planning() method for shared memory mode
                self.global_planner.replan(
                    initial_start=(self.initial_start[0], self.initial_start[1]), 
                    step_T=self.step_T, 
                    RRT_getwaypoint_steps=self.RRT_getwaypoint_steps, 
                    plot_map=True
                )
            else:
                # Use replan() method for ROS message mode
                self.global_planner.replan(
                    initial_start=(self.initial_start[0], self.initial_start[1]), 
                    step_T=self.step_T, 
                    RRT_getwaypoint_steps=self.RRT_getwaypoint_steps, 
                    plot_map=False
                )
            # TODO: Weird
            
            # Publish path
            self.publish_path()
            
            self.last_plan_time = rospy.get_time()
            
        except Exception as e:
            rospy.logerr("[PathPlanner] Error during planning: %s", str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.planning_in_progress = False




####################################### PUBLISHING #######################################

    def publish_path(self):
        """Publish the planned path as a ROS Path message."""
        if len(self.global_planner.path) == 0:
            rospy.logwarn("[PathPlanner] No path to publish")
            return
        
        global_path_msg = Path()
        global_path_msg.header.frame_id = self.frame_id
        global_path_msg.header.stamp = rospy.Time.now()
        
        # Handle different path formats
        for pt in self.global_planner.path:
            pose = PoseStamped()
            pose.header.frame_id = self.frame_id
            pose.header.stamp = global_path_msg.header.stamp
            
            # Handle both tuple/list format and Node object format
            if hasattr(pt, 'x') and hasattr(pt, 'y'):
                # Node object (from shared memory mode)
                pose.pose.position.x = pt.x
                pose.pose.position.y = pt.y
            else:
                # Tuple/list format (from ROS message mode)
                pose.pose.position.x = pt[0]
                pose.pose.position.y = pt[1]
            
            pose.pose.position.z = 1.0
            pose.pose.orientation.w = 1.0
            global_path_msg.poses.append(pose)
        
        self.global_path_pub.publish(global_path_msg)
        # rospy.loginfo("Published path with %d waypoints", len(self.global_planner.path))
    






####################################### CALLBACKS #######################################

    def pose_callback(self, msg):
        """Update robot pose from PoseStamped message."""
        p = msg.pose.position
        q = msg.pose.orientation
        
        yaw = np.arctan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )
        
        if self.use_shared_memory:
            # Update pose in shared memory (via backend)
            self.global_map.backend.set_robot_pose(p.x, p.y, wrap_to_pi(yaw))
            # Also update local copy
            self.global_map.update_robot_pose()
        else:
            # Update local pose (ROS message mode)
            self.global_map.robot_x = p.x
            self.global_map.robot_y = p.y
            self.global_map.robot_heading = wrap_to_pi(yaw)
    
    def replanning_callback(self, msg):
        """Callback for replanning signal via ROS topic"""
        if msg.data:
            self.replanning_needed = True
            rospy.loginfo("[PathPlanner] Replanning requested via ROS topic")


    
    def map_callback(self, msg):
        """Handle incoming map message and trigger planning (ROS message mode only)."""
        if self.use_shared_memory:
            rospy.logwarn("[PathPlanner] map_callback called in shared memory mode. This should not happen.")
            return
        
        if self.planning_in_progress:
            rospy.logwarn_throttle(1.0, "[PathPlanner] Planning in progress, skipping map update")
            return
        
        if rospy.get_time() < self.initialization_time:
            return
        
        # Update map data using helper functions
        success = False
        if self.use_gridmap_msg:
            success = populate_map_from_gridmap(self.global_map, msg)
        else:
            success = populate_map_from_float32multiarray(self.global_map, msg)
        
        if not success:
            rospy.logwarn("[PathPlanner] Failed to update map from message")
            return

        # Trigger planning for each map update
        self.plan_and_publish()
    
    





####################################### RUN #######################################

    def run(self):
        if self.use_shared_memory:
            rospy.loginfo("[PathPlanner] Planning Node started (shared memory mode). Waiting for maps and replanning signals...")
            planning_rate = rospy.Rate(10)  # 10 Hz for shared memory mode
            
            while not rospy.is_shutdown():
                # Update map flags from shared memory (for reader mode)
                if hasattr(self.global_map, 'update_map_flags'):
                    self.global_map.update_map_flags()
                # Check if maps are ready
                if not self.global_map.is_TraversabilityMap_built:
                    rospy.loginfo_throttle(5, "[PathPlanner] Waiting for maps to be built...")
                    planning_rate.sleep()
                    continue

                # Check for replanning requests via ROS topic
                # replanning_needed is set by replanning_callback when /replanning_signal is received
                if self.replanning_needed and not self.planning_in_progress:
                    self.plan_and_publish()
                    self.replanning_needed = False

                planning_rate.sleep()
        else:
            rospy.loginfo("[PathPlanner] Planning Node started (ROS message mode). Waiting for map updates...")
            planning_rate = rospy.Rate(1)  # 1 Hz rate
            
            while not rospy.is_shutdown():
                # plan_and_publish is called in map_callback for ROS message mode
                # We don't call it here to avoid threading and locking issues
                planning_rate.sleep()

def main():
    try:
        node = PlanningNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.signal_shutdown("Finished execution")


if __name__ == '__main__':
    main()