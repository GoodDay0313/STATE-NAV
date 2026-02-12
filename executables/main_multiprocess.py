#!/usr/bin/env python3
"""
Parent Process - Spawns WorldModel and Planner as separate processes

Architecture:
- Parent process creates shared resources (mp.Value, mp.Lock, SharedMemory)
- Spawns child processes (WorldModel writer, Planner reader)
- Each child runs its own ROS node
- Children share memory via inherited objects
- True parallel execution (no context switching overhead)

Usage:
    ros2 run statenav_global multiprocess
"""

import multiprocessing as mp
import signal
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Import from main_global_planning
# Note: Import happens inside worker functions to avoid ROS initialization issues


class SharedMetadata:
    """
    Shared metadata structure for inter-process communication.
    Must be defined at module level for multiprocessing pickling.
    """
    def __init__(self):
        # Version tracking (atomic)
        self.version = mp.Value('i', 0)
        
        # Map state flags (atomic)
        self.is_trav_map_built = mp.Value('b', False)
        self.is_elev_map_built = mp.Value('b', False)
        
        # Map parameters (set once, then read-only)
        self.map_resolution = mp.Value('d', 0.0)
        self.elev_map_rows = mp.Value('i', 0)
        self.elev_map_cols = mp.Value('i', 0)
        self.trav_map_rows = mp.Value('i', 0)
        self.trav_map_cols = mp.Value('i', 0)
        self.trav_map_layers = mp.Value('i', 0)
        
        # Map bounds
        self.elev_map_xmin = mp.Value('d', 0.0)
        self.elev_map_xmax = mp.Value('d', 0.0)
        self.elev_map_ymin = mp.Value('d', 0.0)
        self.elev_map_ymax = mp.Value('d', 0.0)
        
        # Robot state (updated frequently)
        self.robot_x = mp.Value('d', 0.0)
        self.robot_y = mp.Value('d', 0.0)
        self.robot_heading = mp.Value('d', 0.0)
        
        # Locks (shared across processes!)
        self.write_lock = mp.Lock()  # Exclusive: Only ONE writer at a time
        self.read_lock = mp.RLock()  # Shared: Multiple readers can acquire simultaneously


def world_model_worker(shared_metadata, shared_backend_config):
    """
    Worker function for WorldModel process (writer)

    Args:
        shared_metadata: SharedMetadata object (mp.Value, mp.Lock) - shared with planner
        shared_backend_config: Dict with backend configuration
    """
    import rclpy
    from executables.main_worldmodel import WorldModelNode

    # Initialize ROS2 in THIS process
    rclpy.init()
    world_model_node = None  # Initialize to handle creation failures

    try:
        # Create WorldModelNode with shared metadata
        # Note: ROS2 nodes always initialize via __init__, no init_ros parameter
        world_model_node = WorldModelNode(shared_metadata=shared_metadata)
        world_model_node.get_logger().info(f"[WorldModel] WorldModel process started (PID: {mp.current_process().pid})")

        # Run world model node
        world_model_node.run()
    finally:
        if world_model_node is not None:
            world_model_node.get_logger().info("[WorldModel] WorldModel process shutting down")
        if rclpy.ok():
            rclpy.shutdown()


def planner_worker(shared_metadata, shared_backend_config):
    """
    Worker function for Planner process (reader)

    Args:
        shared_metadata: SharedMetadata object (mp.Value, mp.Lock) - shared with writer
        shared_backend_config: Dict with backend configuration
    """
    import rclpy
    from executables.main_global_planning import PlanningNode

    # Initialize ROS2 in THIS process
    rclpy.init()
    planning_node = None  # Initialize to handle creation failures

    try:
        # Create PlanningNode with shared metadata
        # Note: ROS2 nodes always initialize via __init__, no init_ros parameter
        planning_node = PlanningNode(shared_metadata=shared_metadata)
        planning_node.get_logger().info(f"[Planner] Planner process started (PID: {mp.current_process().pid})")

        # Run planner node
        planning_node.run()
    finally:
        if planning_node is not None:
            planning_node.get_logger().info("[Planner] Planner process shutting down")
        if rclpy.ok():
            rclpy.shutdown()


def create_shared_metadata():
    """
    Create shared metadata structure in parent process.
    Children will inherit these objects.
    """
    return SharedMetadata()


def signal_handler(signum, frame):
    """Handle signals to gracefully shutdown children"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """
    Parent process: Creates shared resources and spawns children
    """
    # Set multiprocessing start method (important for ROS compatibility)
    # 'spawn' works on all platforms, 'fork' is faster on Unix but can cause issues with ROS
    mp.set_start_method('spawn', force=True)
    
    print("=" * 80)
    print("Parent Process: Creating shared resources and spawning children")
    print("=" * 80)
    
    # Create shared metadata in parent (children will inherit)
    shared_metadata = create_shared_metadata()
    shared_backend_config = {}  # Can pass config if needed
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Spawn child processes
    print("Spawning WorldModel process (writer)...")
    world_model_process = mp.Process(
        target=world_model_worker,
        args=(shared_metadata, shared_backend_config),
        name='WorldModel'
    )
    
    print("Spawning Planner process (reader)...")
    planner_process = mp.Process(
        target=planner_worker,
        args=(shared_metadata, shared_backend_config),
        name='Planner'
    )
    
    # Start processes (they run in parallel!)
    world_model_process.start()
    planner_process.start()
    
    print(f"WorldModel PID: {world_model_process.pid}")
    print(f"Planner PID: {planner_process.pid}")
    print("\nChildren running in parallel. Press Ctrl+C to stop.\n")
    
    try:
        # Wait for children to finish (or until interrupted)
        world_model_process.join()
        planner_process.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user, terminating children...")
        world_model_process.terminate()
        planner_process.terminate()
        world_model_process.join(timeout=5)
        planner_process.join(timeout=5)
    
    print("Parent process exiting")


if __name__ == '__main__':
    main()
