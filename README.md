# STATE-NAV: Stability-Aware

## Overview

Teaser image

**STATE-NAV** is a learning-based traversability estimation and risk-sensitive velocity-based navigation framework for bipedal robots operating in diverse and uneven environments. It learns a stability-aware representation of terrain by carefully selecting a self-supervised locomotion signal and integrates this knowledge into a hierarchical planning framework for safe and efficient navigation.

### Why STATE-NAV?

- **Avoids brittle terrain heuristics**: No reliance on hand-tuned cost functions that break in new environments
- **Robot-specific and environment-agnostic**: Provides a generalizable way to measure traversability tailored to bipedal locomotion
- **Safe learning-planning integration**: Offers a clean abstraction that integrates learning and planning without sacrificing safety
- **Modular and extensible**: Allows researchers to plug in other instability metrics, controllers, or planners

### Key Features

- **Traversability Estimation**: TravFormer, a transformer-based neural network, predicts bipedal instability along with its uncertainty.
Traversability is defined as a stability-aware command velocity: the fastest command that keeps predicted instability within a user-specified limit.
The model supports risk-aware planning via Value at Risk (VaR) and operates directly from elevation maps.
- **TravRRT Global Planning**: A global planner that leverages the predicted stability-aware velocity to generate time-efficient and risk-sensitive paths.
Consistently avoids unsafe terrain without manual weight tuning or environment-specific adjustments.

## Citing

If you use STATE-NAV in your research, please cite:

```bibtex
@article{yoon2025state,
  title={STATE-NAV: Stability-Aware Traversability Estimation for Bipedal Navigation on Rough Terrain},
  author={Yoon, Ziwon and Zhu, Lawrence Y and Gan, Lu and Zhao, Ye},
  journal={arXiv preprint arXiv:2506.01046},
  year={2025}
}
```

## Quick instructions to run

### Cloning Repositories

First, create and navigate into your ROS2 workspace source directory. This ensures the repository is placed in the correct workspace structure.

Second, clone the repo.

```zsh
# Go to your ROS2 workspace (replace with your actual path)
cd $(path_to_your_ros2_workspace)

# Create workspace folder structure
mkdir -p statenav_ws/src

# Move into src directory
cd statenav_ws/src

# Clone repository into state_nav folder
git clone

# Done: repo is now in statenav_ws/src/state_nav
```

### Docker and NVIDIA environment Installation

First, Install Docker: [https://docs.docker.com/desktop/install/ubuntu/](https://docs.docker.com/desktop/install/ubuntu/)
Second, install nvidia container toolkit: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Building Docker Image

An easy way is to download prebuilt container from `docker.io/twoleggedcoffeedrinker/state_nav:latest`.

Or, you can build by your own from scratch.

```zsh
cd $(path_to_your_ros2_workspace)/statenav_ws/src/state_nav
docker build -f docker/Dockerfile.x64 -t biped_nav_sim .
# You might need to change the image name in `runfrombuild.sh`
```

### Opening a container

After building, open a container.

- When reopening container throws an error for XAUTH, delete your /tmp/.X11-unix and /tmp/.docker.xauth and then running ./runfrombuild would work. Running ./runfrombuild with sudo might potentially cause a problem for setting home directory.

```zsh
sudo rm -rf /tmp/.X11-unix
sudo rm -rf /tmp/.docker.xauth
```

```zsh
cd $(path_to_your_ros2_workspace)/statenav_ws/src/state_nav/docker
./runfrombuild.sh
```

### Opening shells and running the Startup script

After opening an container, open an interactive shell in the container.

⚠️ Caution: Before running the script, edit `docker/startup.sh` and set `STATENAV_SRC` to where you cloned this repo 

```zsh
# As seen inside the container
# Example
STATENAV_SRC="${STATENAV_SRC:-${HOST_HOME_DIR}/Desktop/ros2_ws/statenav_ws/src/state_nav}"
```
Run the following. You should be in the container's filesystem.

```zsh
. $HOST_HOME_DIR/$(path_to_your_ros2_workspace)/statenav_ws/src/state_nav/docker/startup.sh
```

This will build ROS2 package. Now you are good to go.
Keep in mind that this build is only valid in this specific container. If you open a new container, you have to do the same thing again.
Also, when you do ROS2 build in a docker container, it might conflict if you do ROS2 build outside of the container.

## Running

Run the followings in separate shells.

```bash
. $HOST_HOME_DIR/$(path_to_your_ros2_workspace)/statenav_ws/src/state_nav/docker/startup.sh
source $HOST_HOME_DIR/$(path_to_your_ros2_workspace)/statenav_ws/install/setup.bash
```

### Step 1 — Set your goal and map settings

Edit `statenav_global/configs/planning_config.yaml`.

Most users only need to change `global_goal` (where to go) and `initial_start` (where to start).
The configuration file allows users to change various parameters related to traversability mapping.

### Step 2 — Build the safe-terrain map (first terminal)

```bash
ros2 run statenav_global traversability_estimation
```

**What it does:** 

```bash
# Launches the traversability estimation node that converts elevation map to traversability map.
# Computes stability-aware traversability maps using the TravFormer neural network.
# This node publishes costmaps that indicate safe navigation regions for the bipedal robot.
```

**Remember:** Start this **before** Step 3. The planner needs this map first.

### Step 3 — Plan a path (second terminal)

```bash
ros2 run statenav_global global_planning
```

**What it does:**

```bash
# Launches the global path planning node that uses TravRRT* to compute optimal paths
# based on the traversability maps. This node subscribes to costmaps and robot pose,
# then publishes planned paths for navigation.
```

### Step 4 — Play recorded data

```bash
ros2 bag play $(path_to_your_ros2_workspace)/src/state_nav/ROS/1017_2.bag
```

**What it does:**

```bash
# Plays back recorded sensor data (camera pointcloud, elevation map, path plan, etc.) from a rosbag file.
# It is recording of one of our outdoor experiments.
# Note: ROS1 bags need to be converted to ROS2 format first using ros1_bridge.
# Replace $(path_to_your_ros2_workspace) with your actual ROS2 workspace path.
```

### Step 5 — Visualize in RViz

```bash
rviz2 -d $(path_to_your_ros2_workspace)/src/state_nav/ROS/rviz_setting.rviz
```

**What it does:**

```bash
# Launches RViz2 visualization tool with a pre-configured setup to visualize
# the traversability maps, planned paths, robot pose, and other ROS2 topics.
# Note: The RViz config file may need to be updated for ROS2 compatibility.
# Replace $(path_to_your_ros2_workspace) with your actual ROS2 workspace path.
```

### Quick reference


| Step | Command                     | What you get              |
| ---- | --------------------------- | ------------------------- |
| 1    | Edit `planning_config.yaml` | Start, goal, and settings |
| 2    | `traversability_estimation` | Safe-terrain map          |
| 3    | `global_planning`           | Path to goal planning     |
| 4    | `ros2 bag play ...`         | Demo replay               |
| 5    | `rviz2 -d ...`              | Visualization             |


## Acknowledgments

This project builds upon the following open-source works:

### Docker and RViz Configuration

The Docker setup and RViz visualization configurations are adapted from [elevation_mapping_cupy](https://github.com/leggedrobotics/elevation_mapping_cupy) ([MIT License](https://github.com/leggedrobotics/elevation_mapping_cupy/blob/main/LICENSE)).

### RRT* Path Planning

The RRT* (Rapidly-exploring Random Tree Star) implementation is based on [pypolo](https://github.com/Weizhe-Chen/pypolo) ([MIT License](https://github.com/Weizhe-Chen/pypolo/blob/main/LICENSE)), a Python library for Robotic Information Gathering.

We gratefully acknowledge the contributions of these projects to the robotics community.