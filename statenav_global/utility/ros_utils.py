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


def publish_costmap_float32multiarray(global_map, frame_id, global_costmap_pub, node=None):
    """
    Publish costmap using Float32MultiArray (original method).

    Args:
        global_map: Map instance (BaseMap or subclass)
        frame_id: ROS frame ID
        global_costmap_pub: ROS2 publisher for costmap
        node: ROS2 node instance (optional, for logging)
    """
    # Full-map grid index bounds
    row_xmin, row_xmax = global_map.map_size_rows - 1, 0
    col_ymin, col_ymax = global_map.map_size_cols - 1, 0
    xmin_p, ymin_p = global_map.map_xmin, global_map.map_ymin
    xmax_p = row_xmin * global_map.map_resolution + xmin_p + 1e-4
    ymax_p = col_ymin * global_map.map_resolution + ymin_p + 1e-4

    msg = Float32MultiArray()
    # Metadata
    msg.data.extend([
        global_map.map_resolution, global_map.TraversabilityMap_theta_resolution,
        xmin_p, xmax_p, ymin_p, ymax_p,
        global_map.TraversabilityMap_theta_min, global_map.TraversabilityMap_theta_max
    ])
    # Data
    for i in range(0, global_map.map_size_rows):
        for j in range(0, global_map.map_size_cols):
            for k in range(global_map.TraversabilityMap_size_layers):
                msg.data.extend(global_map.TraversabilityMap[i, j, k, :])
    
    global_costmap_pub.publish(msg)


def publish_costmap_gridmap(global_map, frame_id, global_costmap_pub, node=None):
    """
    Publish costmap using GridMap message type.

    Args:
        global_map: Map instance (BaseMap or subclass)
        frame_id: ROS frame ID
        global_costmap_pub: ROS2 publisher for costmap
        node: ROS2 node instance (required for time stamps)
    """
    # Full-map grid index bounds
    row_xmin, row_xmax = global_map.map_size_rows - 1, 0
    col_ymin, col_ymax = global_map.map_size_cols - 1, 0
    xmin_p, ymin_p = global_map.map_xmin, global_map.map_ymin
    xmax_p = row_xmin * global_map.map_resolution + xmin_p + 1e-4
    ymax_p = col_ymin * global_map.map_resolution + ymin_p + 1e-4

    num_rows, num_cols = row_xmin, col_ymin
    
    grid_map_msg = GridMap()
    grid_map_msg.header.frame_id = frame_id
    if node is not None:
        grid_map_msg.header.stamp = node.get_clock().now().to_msg()
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
        global_map.map_xmin = data[idx]; idx += 1
        global_map.map_xmax = data[idx]; idx += 1
        global_map.map_ymin = data[idx]; idx += 1
        global_map.map_ymax = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_min = data[idx]; idx += 1
        global_map.TraversabilityMap_theta_max = data[idx]; idx += 1
        
        # Calculate dimensions
        global_map.map_size_rows = int((global_map.map_xmax - global_map.map_xmin) / global_map.map_resolution)
        global_map.map_size_cols = int((global_map.map_ymax - global_map.map_ymin) / global_map.map_resolution)
        global_map.TraversabilityMap_size_layers = int(2 * np.round(np.pi / global_map.TraversabilityMap_theta_resolution))

    else:
        idx = 8 # Skip the metadata but keep the data index
        
    # Calculate expected data size
    expected_size = global_map.map_size_rows * global_map.map_size_cols * global_map.TraversabilityMap_size_layers * 4
    actual_size = len(data) - idx
    
    if actual_size < expected_size:
        _log_warning(f"[PathPlanner] Map data incomplete. Expected {expected_size}, got {actual_size}", node)
        return False
    
    # Reshape data into 4D array [rows, cols, theta_layers, 4_channels]
    map_data = np.array(data[idx:idx+expected_size], dtype=np.float32)
    global_map.TraversabilityMap = map_data.reshape(
        (global_map.map_size_rows, 
         global_map.map_size_cols, 
         global_map.TraversabilityMap_size_layers, 
         4)
    )
    
    global_map.is_TraversabilityMap_built = True # Avoiding unnecessary initialization of the map. Assuming the map metadata does not change.
    # _log_info(f"Map updated: {global_map.map_size_rows}x{global_map.map_size_cols}x{global_map.TraversabilityMap_size_layers}", node)
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
    
    global_map.map_xmin = center_x - length_x / 2.0
    global_map.map_xmax = center_x + length_x / 2.0
    global_map.map_ymin = center_y - length_y / 2.0
    global_map.map_ymax = center_y + length_y / 2.0
    
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
    
    global_map.map_size_rows = num_rows
    global_map.map_size_cols = num_cols
    
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


def Rviz_vis_travmap(global_map, pub, node, frame_id):
    """Publish elevation + north-direction cmd_v as a GridMap for Rviz2 via grid_map_rviz_plugin.
    Intended to be called at low frequency (<=0.5 Hz). NaN cells are rendered as empty."""
    import array as _array

    if not global_map.is_TraversabilityMap_built:
        return

    # Spatial dimensions of the global map grid
    num_rows = global_map.map_size_rows  # x-axis extent in cells
    num_cols = global_map.map_size_cols  # y-axis extent in cells

    # Lazy-allocate flat float32 buffers on first call (one per layer); safe since map
    # dimensions are fixed by the time is_TraversabilityMap_built is True
    if global_map._vis_flat_bufs is None:
        _n = num_rows * num_cols
        global_map._vis_flat_bufs = [np.empty(_n, dtype=np.float32),   # elevation layer [num_rows*num_cols]
                                      np.empty(_n, dtype=np.float32)]    # cmd_v layer     [num_rows*num_cols]

    # Theta layer index for north (0 deg): idx = (0 - theta_min) / theta_resolution
    north_theta_idx = int(round((0.0 - global_map.TraversabilityMap_theta_min) / global_map.TraversabilityMap_theta_resolution))

    # Build GridMap header and spatial metadata
    grid_map_msg = GridMap()
    grid_map_msg.header.frame_id = frame_id
    grid_map_msg.header.stamp = node.get_clock().now().to_msg()
    grid_map_msg.info.resolution = float(global_map.map_resolution)
    grid_map_msg.info.length_x = float(global_map.map_xmax - global_map.map_xmin)
    grid_map_msg.info.length_y = float(global_map.map_ymax - global_map.map_ymin)
    grid_map_msg.info.pose.position.x = float((global_map.map_xmin + global_map.map_xmax) / 2.0)
    grid_map_msg.info.pose.position.y = float((global_map.map_ymin + global_map.map_ymax) / 2.0)
    grid_map_msg.info.pose.position.z = 0.0
    grid_map_msg.info.pose.orientation.w = 1.0

    # Layers: (layer_name, 2D array of shape [num_rows, num_cols])
    layers_to_publish = [
        ("elevation", global_map.ElevationMap),                                  # global elevation grid [num_rows, num_cols]
        ("cmd_v",     global_map.TraversabilityMap[:, :, north_theta_idx, 0]),   # cmd_v at north heading [num_rows, num_cols]
    ]

    for layer_idx, (layer_name, data_array) in enumerate(layers_to_publish):
        grid_map_msg.layers.append(layer_name)

        layer_data = Float32MultiArray()
        layout = MultiArrayLayout()

        # Column-major layout: outer dim = cols (y-axis), inner dim = rows (x-axis)
        # Required by grid_map convention for correct Rviz2 display
        dim_col = MultiArrayDimension()
        dim_col.label = "column_index"
        dim_col.size = num_cols
        dim_col.stride = num_rows

        dim_row = MultiArrayDimension()
        dim_row.label = "row_index"
        dim_row.size = num_rows
        dim_row.stride = 1

        layout.dim = [dim_col, dim_row]
        layout.data_offset = 0
        layer_data.layout = layout

        # Fill pre-allocated buffer in column-major order via a single typed copy:
        # reshape(num_cols, num_rows) makes the flat buffer a view with shape [num_cols, num_rows],
        # then copying data_array.T (shape [num_cols, num_rows]) writes buf[j*num_rows+i]=data[i,j].
        # This avoids intermediate ravel/astype allocations. NaN values pass through unchanged.
        np.copyto(global_map._vis_flat_bufs[layer_idx].reshape(num_cols, num_rows), data_array.T)
        # array.array avoids creating per-element Python float objects (5-10x faster than tolist())
        layer_data.data = _array.array('f', global_map._vis_flat_bufs[layer_idx])

        grid_map_msg.data.append(layer_data)

    pub.publish(grid_map_msg)

