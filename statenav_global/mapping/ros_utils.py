"""
ROS2 utilities for mapping module.
Contains ROS2 message publishing functions for traversability maps.
"""

import numpy as np
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from grid_map_msgs.msg import GridMap


def _log_warning(msg, node=None):
    """Helper function for logging warnings (compatible with ROS2 or print)."""
    if node is not None:
        node.get_logger().warning(msg)
    else:
        print(f"[WARN] {msg}")


def _log_info(msg, node=None):
    """Helper function for logging info (compatible with ROS2 or print)."""
    if node is not None:
        node.get_logger().info(msg)
    else:
        print(f"[INFO] {msg}")


def _calculate_costmap_region(global_map, pub_locally, global_planner, step_T, horizon_multiplier, MPC_horizon):
    """Calculate the region bounds for costmap publishing."""
    if pub_locally and global_planner is not None:
        cmd_v_limit, cmd_w_limit = global_map.get_cmd_limits()
        waypoint = global_map.get_waypoint(
            global_map.robot_x, global_map.robot_y, global_map.robot_heading,
            step_T, horizon_multiplier * MPC_horizon, cmd_v_limit, cmd_w_limit, global_planner.path
        )
        xmin, xmax = min(global_map.robot_x, waypoint[0]), max(global_map.robot_x, waypoint[0])
        ymin, ymax = min(global_map.robot_y, waypoint[1]), max(global_map.robot_y, waypoint[1])
        padding = 1
    else:
        xmin, xmax = global_map.TraversabilityMap_xmin, global_map.TraversabilityMap_xmax
        ymin, ymax = global_map.TraversabilityMap_ymin, global_map.TraversabilityMap_ymax
        padding = 0
    
    # Apply padding and clamp to map bounds
    xmin_padded = max(xmin - padding, global_map.TraversabilityMap_xmin)
    xmax_padded = min(xmax + padding, global_map.TraversabilityMap_xmax)
    ymin_padded = max(ymin - padding, global_map.TraversabilityMap_ymin)
    ymax_padded = min(ymax + padding, global_map.TraversabilityMap_ymax)
    
    # Convert to grid indices
    row_xmin = int((global_map.TraversabilityMap_xmax - xmin_padded) / global_map.map_resolution)
    row_xmax = int((global_map.TraversabilityMap_xmax - xmax_padded) / global_map.map_resolution)
    col_ymin = int((global_map.TraversabilityMap_ymax - ymin_padded) / global_map.map_resolution)
    col_ymax = int((global_map.TraversabilityMap_ymax - ymax_padded) / global_map.map_resolution)
    
    # Boundary checks
    row_xmin = min(row_xmin, global_map.TraversabilityMap_size_rows - 1)
    row_xmax = max(row_xmax, 0)
    col_ymin = min(col_ymin, global_map.TraversabilityMap_size_cols - 1)
    col_ymax = max(col_ymax, 0)
    
    # Recalculate bounds from clamped indices
    xmax_padded = (row_xmin - row_xmax) * global_map.map_resolution + xmin_padded + 1e-4
    ymax_padded = (col_ymin - col_ymax) * global_map.map_resolution + ymin_padded + 1e-4
    
    return row_xmin, row_xmax, col_ymin, col_ymax, xmin_padded, xmax_padded, ymin_padded, ymax_padded


def publish_costmap_float32multiarray(global_map, frame_id, global_costmap_pub, pub_locally, global_planner,
                                       step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon, node=None):
    """
    Publish costmap using Float32MultiArray (original method).

    Args:
        global_map: Map instance (BaseMap or subclass)
        frame_id: ROS frame ID
        global_costmap_pub: ROS2 publisher for costmap
        pub_locally: If True, publish only local region around robot/waypoint
        global_planner: Optional planner instance for waypoint calculation
        step_T: Time step
        localmap_getwaypoint_horizonmultiplier: Horizon multiplier for waypoint
        MPC_horizon: MPC horizon length
        node: ROS2 node instance (optional, for logging)
    """
    row_xmin, row_xmax, col_ymin, col_ymax, xmin_p, xmax_p, ymin_p, ymax_p = _calculate_costmap_region(
        global_map, pub_locally, global_planner, step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon
    )
    
    msg = Float32MultiArray()
    # Metadata
    msg.data.extend([
        global_map.map_resolution, global_map.TraversabilityMap_theta_resolution,
        xmin_p, xmax_p, ymin_p, ymax_p,
        global_map.TraversabilityMap_theta_min, global_map.TraversabilityMap_theta_max
    ])
    # Data
    for i in range(row_xmax, row_xmin):
        for j in range(col_ymax, col_ymin):
            for k in range(global_map.TraversabilityMap_size_layers):
                msg.data.extend(global_map.TraversabilityMap[i, j, k, :])
    
    global_costmap_pub.publish(msg)


def publish_costmap_gridmap(global_map, frame_id, global_costmap_pub, pub_locally, global_planner,
                            step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon, node=None):
    """
    Publish costmap using GridMap message type.

    Args:
        global_map: Map instance (BaseMap or subclass)
        frame_id: ROS frame ID
        global_costmap_pub: ROS2 publisher for costmap
        pub_locally: If True, publish only local region around robot/waypoint
        global_planner: Optional planner instance for waypoint calculation
        step_T: Time step
        localmap_getwaypoint_horizonmultiplier: Horizon multiplier for waypoint
        MPC_horizon: MPC horizon length
        node: ROS2 node instance (required for time stamps)
    """
    row_xmin, row_xmax, col_ymin, col_ymax, xmin_p, xmax_p, ymin_p, ymax_p = _calculate_costmap_region(
        global_map, pub_locally, global_planner, step_T, localmap_getwaypoint_horizonmultiplier, MPC_horizon
    )
    
    num_rows, num_cols = row_xmin - row_xmax, col_ymin - col_ymax
    
    grid_map_msg = GridMap()
    grid_map_msg.info.header.frame_id = frame_id
    if node is not None:
        grid_map_msg.info.header.stamp = node.get_clock().now().to_msg()
    grid_map_msg.info.resolution = global_map.map_resolution
    grid_map_msg.info.length_x = xmax_p - xmin_p
    grid_map_msg.info.length_y = ymax_p - ymin_p
    grid_map_msg.info.pose.position.x = (xmin_p + xmax_p) / 2.0
    grid_map_msg.info.pose.position.y = (ymin_p + ymax_p) / 2.0
    grid_map_msg.info.pose.position.z = 0.0
    grid_map_msg.info.pose.orientation.w = 1.0
    
    channel_names = ["cmd_v", "cmd_w", "auxiliary_score", "auxiliary_score_std"]
    
    # Create layers for each (theta_layer, channel) combination
    for theta_layer in range(global_map.TraversabilityMap_size_layers):
        for channel_idx, channel_name in enumerate(channel_names):
            layer_name = f"{channel_name}_theta_{theta_layer}"
            grid_map_msg.layers.append(layer_name)
            
            layer_data = Float32MultiArray()
            layout = MultiArrayLayout()
            layout.dim = [
                MultiArrayDimension(label="column_index", size=num_cols, stride=num_rows),
                MultiArrayDimension(label="row_index", size=num_rows, stride=1)
            ]
            layout.data_offset = 0
            layer_data.layout = layout
            
            # Extract data (row-major order)
            for j in range(col_ymax, col_ymin):
                for i in range(row_xmax, row_xmin):
                    layer_data.data.append(float(global_map.TraversabilityMap[i, j, theta_layer, channel_idx]))
            
            grid_map_msg.data.append(layer_data)
    
    # Add metadata layers
    metadata = {
        "theta_resolution": global_map.TraversabilityMap_theta_resolution,
        "theta_min": global_map.TraversabilityMap_theta_min,
        "theta_max": global_map.TraversabilityMap_theta_max,
        "num_theta_layers": float(global_map.TraversabilityMap_size_layers),
        "xmin": xmin_p, "xmax": xmax_p, "ymin": ymin_p, "ymax": ymax_p
    }
    
    for meta_name, meta_value in metadata.items():
        grid_map_msg.layers.append(meta_name)
        meta_layer = Float32MultiArray()
        meta_layer.layout = layout
        meta_layer.data = [float(meta_value)] * (num_rows * num_cols)
        grid_map_msg.data.append(meta_layer)
    
    global_costmap_pub.publish(grid_map_msg)


def populate_map_from_float32multiarray(global_map, msg, node=None):
    """Populate existing map object from Float32MultiArray message."""
    if len(msg.data) < 8:
        _log_warning("[PathPlanner] Map message too short, skipping", node)
        return False
        
    data = msg.data
    idx = 0
        
    if not global_map.is_TraversabilityMap_built: # If the map is not built, extract the metadata
        # Extract metadata
        global_map.map_resolution = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_resolution = data[idx]; idx += 1
        global_map.TraversabilityMap_xmin = data[idx]; idx += 1
        global_map.TraversabilityMap_xmax = data[idx]; idx += 1
        global_map.TraversabilityMap_ymin = data[idx]; idx += 1
        global_map.TraversabilityMap_ymax = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_min = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_max = data[idx]; idx += 1
        
        # Calculate dimensions
        global_map.TraversabilityMap_size_rows = int((global_map.TraversabilityMap_xmax - global_map.TraversabilityMap_xmin) / global_map.map_resolution)
        global_map.TraversabilityMap_size_cols = int((global_map.TraversabilityMap_ymax - global_map.TraversabilityMap_ymin) / global_map.map_resolution)
        global_map.TraversabilityMap_size_layers = int(2 * np.round(np.pi / global_map.TraversabilityMap_theta_resolution))

    else:
        idx = 8 # Skip the metadata but keep the data index
        
    # Calculate expected data size
    expected_size = global_map.TraversabilityMap_size_rows * global_map.TraversabilityMap_size_cols * global_map.TraversabilityMap_size_layers * 4
    actual_size = len(data) - idx
    
    if actual_size < expected_size:
        _log_warning(f"[PathPlanner] Map data incomplete. Expected {expected_size}, got {actual_size}", node)
        return False
    
    # Reshape data into 4D array [rows, cols, theta_layers, 4_channels]
    map_data = np.array(data[idx:idx+expected_size], dtype=np.float32)
    global_map.TraversabilityMap = map_data.reshape(
        (global_map.TraversabilityMap_size_rows, 
         global_map.TraversabilityMap_size_cols, 
         global_map.TraversabilityMap_size_layers, 
         4)
    )
    
    global_map.is_TraversabilityMap_built = True # Avoiding unnecessary initialization of the map. Assuming the map metadata does not change.
    # _log_info(f"Map updated: {global_map.TraversabilityMap_size_rows}x{global_map.TraversabilityMap_size_cols}x{global_map.TraversabilityMap_size_layers}", node)
    return True


def populate_map_from_gridmap(global_map, msg, node=None):
    """Populate existing map object from GridMap message."""
    # Extract metadata from GridMap info
    global_map.map_resolution = msg.info.resolution
    length_x = msg.info.length_x
    length_y = msg.info.length_y
    
    # Calculate map bounds from center position
    center_x = msg.info.pose.position.x
    center_y = msg.info.pose.position.y
    
    global_map.TraversabilityMap_xmin = center_x - length_x / 2.0
    global_map.TraversabilityMap_xmax = center_x + length_x / 2.0
    global_map.TraversabilityMap_ymin = center_y - length_y / 2.0
    global_map.TraversabilityMap_ymax = center_y + length_y / 2.0
    
    # Get dimensions from first layer
    if len(msg.layers) == 0 or len(msg.data) == 0:
        _log_warning("[PathPlanner] GridMap has no layers", node)
        return False

    first_layer = msg.data[0]
    if len(first_layer.layout.dim) < 2:
        _log_warning("[PathPlanner] GridMap layer has insufficient dimensions", node)
        return False
    
    # Note: GridMap uses column_index, row_index order
    num_cols = first_layer.layout.dim[0].size
    num_rows = first_layer.layout.dim[1].size
    
    global_map.TraversabilityMap_size_rows = num_rows
    global_map.TraversabilityMap_size_cols = num_cols
    
    # Extract theta information from metadata layers
    theta_resolution = None
    theta_min = None
    theta_max = None
    num_theta_layers = None
    
    for i, layer_name in enumerate(msg.layers):
        if i < len(msg.data):
            if layer_name == "theta_resolution":
                theta_resolution = msg.data[i].data[0]
            elif layer_name == "theta_min":
                theta_min = msg.data[i].data[0]
            elif layer_name == "theta_max":
                theta_max = msg.data[i].data[0]
            elif layer_name == "num_theta_layers":
                num_theta_layers = int(msg.data[i].data[0])
    
    # If metadata not found, infer from layer names
    if theta_resolution is None:
        theta_indices = set()
        for layer_name in msg.layers:
            if "_theta_" in layer_name:
                try:
                    theta_idx = int(layer_name.split("_theta_")[-1])
                    theta_indices.add(theta_idx)
                except:
                    pass
        
        if len(theta_indices) > 0:
            num_theta_layers = len(theta_indices)
            if theta_min is None:
                theta_min = -np.pi
            if theta_max is None:
                theta_max = np.pi - (np.pi / num_theta_layers)
            if theta_resolution is None:
                theta_resolution = (theta_max - theta_min) / (num_theta_layers - 1) if num_theta_layers > 1 else np.pi / 4
    
    if num_theta_layers is None:
        _log_warning("[PathPlanner] Could not determine number of theta layers, defaulting to 8", node)
        num_theta_layers = 8
        theta_min = -np.pi
        theta_max = np.pi - np.pi/4
        theta_resolution = np.pi / 4
    
    global_map.TraversabilityMap_size_layers = num_theta_layers
    global_map.TraversabilityMap_theta_min = theta_min if theta_min is not None else -np.pi
    global_map.TraversabilityMap_theta_max = theta_max if theta_max is not None else (np.pi - np.pi/4)
    global_map.TraversabilityMap_theta_resolution = theta_resolution if theta_resolution is not None else np.pi/4
    
    # Reconstruct 4D array from GridMap layers
    channel_names = ["cmd_v", "cmd_w", "auxiliary_score", "auxiliary_score_std"]
    
    global_map.TraversabilityMap = np.full(
        (num_rows, num_cols, num_theta_layers, 4), 
        np.nan, 
        dtype=np.float32
    )
    
    layer_idx = 0
    for theta_layer in range(num_theta_layers):
        for channel_idx, channel_name in enumerate(channel_names):
            layer_name = f"{channel_name}_theta_{theta_layer}"
            
            if layer_idx < len(msg.layers) and layer_idx < len(msg.data):
                if msg.layers[layer_idx] == layer_name:
                    layer_data = np.array(msg.data[layer_idx].data, dtype=np.float32)
                    
                    # Reshape from column-major to row-major
                    layer_data_reshaped = layer_data.reshape((num_cols, num_rows)).T
                    
                    global_map.TraversabilityMap[:, :, theta_layer, channel_idx] = layer_data_reshaped
                    layer_idx += 1
                else:
                    _log_warning(f"[PathPlanner] Expected layer {layer_name}, got {msg.layers[layer_idx]}", node)
                    layer_idx += 1

    global_map.is_TraversabilityMap_built = True
    # _log_info(f"Map updated from GridMap: {num_rows}x{num_cols}x{num_theta_layers}", node)
    return True

