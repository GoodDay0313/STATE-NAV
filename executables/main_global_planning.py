#!/usr/bin/env python3
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
# Note: tf2_ros imports available if needed for future TF operations
# from tf2_ros import TransformListener, Buffer, TransformException
# import tf2_geometry_msgs

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
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





































class PlanningNode(Node):
    """Main planning node that subscribes to map and publishes paths."""

    def __init__(self, shared_metadata=None):
        """
        Initialize PlanningNode

        Args:
            shared_metadata: Optional SharedMetadata object from parent process.
                           If provided, uses shared memory with parent-spawning approach.
                           If None, uses config to determine mode (named shared memory or ROS messages).
        """
        super().__init__('planning_node')

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
                self.get_logger().info("[PathPlanner] Initializing in SHARED MEMORY mode (reader, parent-spawning)")
                backend = SharedMemoryBackend(mode='reader', shared_metadata=shared_metadata)
            else:
                self.get_logger().info("[PathPlanner] Initializing in SHARED MEMORY mode (reader, named)")
                backend = SharedMemoryBackend(mode='reader')
            
            # Note: CMDbasedMapReaderProxy only supports CMDbasedMap for now
            # If you need other map types, you'll need to create reader proxies for them
            if self.cfg.trav_option == "Proposed" and self.cfg.planner_option == "safecmd":
                self.global_map = CMDbasedMapReaderProxy(backend)
                # Set goal after initialization
                self.global_map.goal_x = global_goal[0]
                self.global_map.goal_y = global_goal[1]
            else:
                self.get_logger().warning(f"[PathPlanner] Shared memory mode currently only supports CMDbasedMap with safecmd option.")
                self.get_logger().warning(f"[PathPlanner] Requested: {self.cfg.trav_option}/{self.cfg.planner_option}. Falling back to ROS message mode.")
                self.use_shared_memory = False
                self.shared_metadata = None
        
        if not self.use_shared_memory:
            # ROS message mode: create map and subscribe to topics
            self.get_logger().info("[PathPlanner] Initializing in ROS MESSAGE mode")
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
        
        self.get_logger().info(f"[PathPlanner] Map class initialized. Type: {type(self.global_map).__name__}, Mode: {'SHARED MEMORY' if self.use_shared_memory else 'ROS MESSAGE'}")
        
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

        # ROS2 setup
        self.frame_id = self.cfg.get('frame_id', 'map')

        # QoS profile for subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers (different based on mode)
        if self.use_shared_memory:
            # Shared memory mode: subscribe to pose and replanning signal via ROS
            self.create_subscription(PoseStamped, "/robot/pose", self.pose_callback, qos_profile)
            self.create_subscription(Bool, "/replanning_signal", self.replanning_callback, qos_profile)
            self.get_logger().info("[PathPlanner] Subscribed to /robot/pose and /replanning_signal (shared memory mode)")
        else:
            # ROS message mode: subscribe to map topics and pose
            self.use_gridmap_msg = self.cfg.get('use_gridmap_msg', False)
            if self.use_gridmap_msg:
                self.create_subscription(GridMap, "/global_costmap", self.map_callback, qos_profile)
                self.get_logger().info("[PathPlanner] Subscribed to /global_costmap (GridMap)")
            else:
                self.create_subscription(Float32MultiArray, "/global_costmap", self.map_callback, qos_profile)
                self.get_logger().info("[PathPlanner] Subscribed to /global_costmap (Float32MultiArray)")

            self.create_subscription(PoseStamped, "/robot/pose", self.pose_callback, qos_profile)
            self.get_logger().info("[PathPlanner] Subscribed to /robot/pose")

        print(self.global_map.robot_heading)

        # Publishers
        self.global_path_pub = self.create_publisher(Path, "/global_path", qos_profile)
        self.get_logger().info("[PathPlanner] Publisher initialized: /global_path")
        
        # For getting next waypoint for resetting the tree
        self.step_T = self.cfg.step_T
        self.RRT_getwaypoint_steps = self.cfg.RRT_getwaypoint_steps
        
        # State
        self.last_plan_time = 0
        self.planning_in_progress = False
        self.initialization_time = self.cfg.initialization_time
        self.replanning_needed = True  # For shared memory mode
        # TODO: Handle replanning needed later
        
        self.get_logger().info(f"[PathPlanner] Planning Node initialized with {self.cfg.trav_option}/{self.cfg.planner_option} map type!")
    





    def plan_and_publish(self):
        """Run RRT planning and publish the path."""
        if not self.global_map.is_TraversabilityMap_built:
            self.get_logger().warning("[PathPlanner] Map not built yet, cannot plan")
            return
        
        self.planning_in_progress = True
        self.get_logger().info("[PathPlanner] Planning in progress")
        
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

            self.last_plan_time = self.get_clock().now().nanoseconds / 1e9

        except Exception as e:
            self.get_logger().error("[PathPlanner] Error during planning: %s" % str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.planning_in_progress = False




####################################### PUBLISHING #######################################

    def publish_path(self):
        """Publish the planned path as a ROS Path message."""
        if len(self.global_planner.path) == 0:
            self.get_logger().warning("[PathPlanner] No path to publish")
            return
        
        global_path_msg = Path()
        global_path_msg.header.frame_id = self.frame_id
        global_path_msg.header.stamp = self.get_clock().now().to_msg()
        
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
        # self.get_logger().info("Published path with %d waypoints", len(self.global_planner.path))
    






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
            self.get_logger().info("[PathPlanner] Replanning requested via ROS topic")


    
    def map_callback(self, msg):
        """Handle incoming map message and trigger planning (ROS message mode only)."""
        if self.use_shared_memory:
            self.get_logger().warning("[PathPlanner] map_callback called in shared memory mode. This should not happen.")
            return
        
        if self.planning_in_progress:
            # Note: ROS2 throttling is done differently - using timer-based approach or manual check
            if not hasattr(self, '_last_warning_time'):
                self._last_warning_time = 0.0
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_warning_time > 1.0:
                self.get_logger().warning("[PathPlanner] Planning in progress, skipping map update")
                self._last_warning_time = current_time
            return

        if self.get_clock().now().nanoseconds / 1e9 < self.initialization_time:
            return
        
        # Update map data using helper functions
        success = False
        if self.use_gridmap_msg:
            success = populate_map_from_gridmap(self.global_map, msg, node=self)
        else:
            success = populate_map_from_float32multiarray(self.global_map, msg, node=self)
        
        if not success:
            self.get_logger().warning("[PathPlanner] Failed to update map from message")
            return

        # Trigger planning for each map update
        self.plan_and_publish()
    
    





####################################### RUN #######################################

    def run(self):
        if self.use_shared_memory:
            self.get_logger().info("[PathPlanner] Planning Node started (shared memory mode). Waiting for maps and replanning signals...")
            planning_rate = self.create_rate(10)  # 10 Hz for shared memory mode

            # Initialize throttle timer for logging
            self._last_wait_log_time = 0.0

            while rclpy.ok():
                # Update map flags from shared memory (for reader mode)
                if hasattr(self.global_map, 'update_map_flags'):
                    self.global_map.update_map_flags()
                # Check if maps are ready
                if not self.global_map.is_TraversabilityMap_built:
                    current_time = self.get_clock().now().nanoseconds / 1e9
                    if current_time - self._last_wait_log_time > 5.0:
                        self.get_logger().info("[PathPlanner] Waiting for maps to be built...")
                        self._last_wait_log_time = current_time
                    planning_rate.sleep()
                    continue

                # Check for replanning requests via ROS topic
                # replanning_needed is set by replanning_callback when /replanning_signal is received
                if self.replanning_needed and not self.planning_in_progress:
                    self.plan_and_publish()
                    self.replanning_needed = False

                planning_rate.sleep()
        else:
            self.get_logger().info("[PathPlanner] Planning Node started (ROS message mode). Waiting for map updates...")
            # In ROS2, we typically use rclpy.spin() instead of a manual rate loop
            # plan_and_publish is called in map_callback for ROS message mode

def main():
    rclpy.init()
    try:
        node = PlanningNode()
        if node.use_shared_memory:
            # For shared memory mode, run custom loop
            node.run()
        else:
            # For ROS message mode, use rclpy.spin()
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()