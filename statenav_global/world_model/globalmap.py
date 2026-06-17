from time import time
import numpy as np
import os
import datetime
import math
import random
import threading
import matplotlib
matplotlib.use('QtAgg')  # Use Qt backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from enum import IntEnum
import hydra
from omegaconf import DictConfig, OmegaConf


import torch


from statenav_global.utility import Utils
from statenav_global.utility.utils import get_source_inference_dir





from std_msgs.msg import Float32MultiArray
from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import PoseStamped


import tf_transformations as tf_trans
import transforms3d as tf3
from scipy.ndimage import map_coordinates
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)




from pathlib import Path
_PKG_ROOT = Path(__file__).resolve().parent.parent
cfg = OmegaConf.load(_PKG_ROOT / "configs/planning_config.yaml")


    
step_T = cfg.step_T
MPC_horizon = cfg.MPC_horizon
sampling_dist = cfg.sampling_dist
lookahead_distance = cfg.lookahead_distance

safecmd_sampling_dist = cfg.safecmd_sampling_dist
lookahead_angle = cfg.lookahead_angle

instability_std_multiplier = cfg.instability_std_multiplier





# Check if Times New Roman font is available, fallback to default serif if not
import os
font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
if os.path.exists(font_path):
    times_new_roman = fm.FontProperties(fname=font_path)
    use_times = True
else:
    # Rebuild font cache to ensure matplotlib doesn't try to use unavailable fonts
    fm._load_fontmanager(try_read_cache=False)
    times_new_roman = fm.FontProperties(family='serif')
    use_times = False

FS_TICK: int = 15
FS_LABEL: int = 15
FS_LEGEND: int = 15
FS_TITLE: int = 20
PLOT_DPI: int=1200
PLOT_FORMAT: str='pdf'
RC_PARAMS: dict = {
    # Set background and border settings
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 2,
    'xtick.color': 'black',
    'ytick.color': 'black',
    "font.family": 'serif',
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Nimbus Roman"],  # Use available system fonts
}
savefigureonce_elevmap = cfg.savemap
savefigureonce_travmap = cfg.savemap





# batch putting of local elev map to global elev map
# 



class ElevmapInfo_IntEnum(IntEnum):

    # Lower number means more primitive status
    # Higher number implies lower number status

    ELEV_NotInitialized = 0
    ELEV_MEASURED = 1

class TravmapInfo_IntEnum(IntEnum):

    TRAV_NotInitialized = 0
    TRAV_NEEDUPDATE = 1
    TRAV_Estimated = 2




















class BaseMap:
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_Travformer = True):

        self.debug_print = cfg.debug_print

        # Throttle counter for debug printouts (prints every 50 calls to avoid log flooding)
        self._dbg_callback_count = 0
        self._dbg_update_count = 0

        self.goal_x = goal_x
        self.goal_y = goal_y


        # Robocentric Local Elevation Map from ROS
        self.is_RobocentricMap_received = False
        self.layer_data = {}
        self._map_data_lock = threading.Lock()

        self.RobocentricMap_CenterPosOffset_Worldfr = None # Not necessarily identical to the robot pos
        self.RobocentricMap_CenterRotOffset_Worldfr = None
        
        self.map_resolution = None # shared across all maps
        self.RobocentricMap_LengthY = None
        self.RobocentricMap_LengthX = None
        self.RobocentricMap_Rows = None
        self.RobocentricMap_Cols = None
 
        # Robot position and heading in the world frame
        self.robot_x = cfg.initial_start[0] # initialized
        self.robot_y = cfg.initial_start[1]
        self.robot_roll = 0.0  # initialized
        self.robot_pitch = 0.0  # initialized
        self.robot_heading = cfg.heading_start


        # World frame Global Elevation Map
        self.is_ElevationMap_built = False
        self.ElevationMap = None
        self.map_xmin = env_xmin
        self.map_xmax = env_xmax
        self.map_size_x = self.map_xmax - self.map_xmin
        self.map_size_rows = None

        self.map_ymin = env_ymin
        self.map_ymax = env_ymax
        self.map_size_y = self.map_ymax - self.map_ymin
        self.map_size_cols = None

        # ElevationMap update methods
        self.which_layer = which_layer
        self.mean_over_maps = cfg.mean_over_maps # Boolean. If true, the first layer is used for data and the second layer is used for deciding if it is NaN


        # 3D (x,y,theta(=heading angle)) Traversability map that contains the pre-estimated cmd v and w. World frame
        self.is_TraversabilityMap_built = False
        self.TraversabilityMap = None
        self.TraversabilityMap_theta_min = -np.pi
        self.TraversabilityMap_theta_resolution = np.pi/4
        self.TraversabilityMap_theta_max = np.pi - self.TraversabilityMap_theta_resolution
        self.TraversabilityMap_size_layers = 2 * int(np.round(np.pi / self.TraversabilityMap_theta_resolution))
        print("Traversability Map size layers: ", self.TraversabilityMap_size_layers)




        # Trav and Velocity


        self.InitialGuess_cmd_v = None
        self.InitialGuess_cmd_w = None

        self.InitialGuess_auxiliary_score = None
        self.InitialGuess_auxiliary_score_std = None


        self.maximum_cmd_v = cfg.maximum_cmd_v
        self.maximum_cmd_w = cfg.maximum_cmd_w





        
        

        # TraversabilityMap update methods
        self.TravUpdate_XYResolution = preest_update_resolution
        self.TravUpdate_Pts = np.empty((0, 2))

        self.elevation_diff_threshold = cfg.elevation_diff_threshold
        self.RandomRate_TravUpdate = cfg.RandomRate_TravUpdate


        self.ElevmapInfo_IntEnum = ElevmapInfo_IntEnum
        self.is_ElevMap_MetaInfo_built = False
        self.ElevMap_MetaInfo = None

        self.TravmapInfo_IntEnum = TravmapInfo_IntEnum
        self.is_TravMap_MetaInfo_built = False
        self.TravMap_MetaInfo = None
        
        self.TravMap_MetaInfo_theta_min = self.TraversabilityMap_theta_min
        self.TravMap_MetaInfo_theta_resolution = self.TraversabilityMap_theta_resolution
        self.TravMap_MetaInfo_theta_max = self.TraversabilityMap_theta_max
        self.TravMap_MetaInfo_size_layers = self.TraversabilityMap_size_layers



        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = 0.5
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Traversability Score"

        # Pre-allocated flat float32 buffers for ros_utils.Rviz_vis_travmap (one per layer)
        # Allocated in Initilize_map once map dimensions are known; shape: [num_rows * num_cols]
        self._vis_flat_bufs = None


        # ML related
        self.device = 'cuda'

        # Travformer related
        self.local_patch_size = cfg.local_patch_size  # in meters
        self.elevonly_vw2instab_model = None

        def _load_model(checkpoint_path, model_type='Travformer', device='cpu'):

            from statenav_global.inference.uncertainty_models import ElevationOnlyNetworkMLL
            if model_type == 'Travformer':
                model = ElevationOnlyNetworkMLL()

            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            return model


        if load_Travformer:
            inference_dir = get_source_inference_dir()
            elevonly_vw2instab_model_checkpoint = os.path.join(inference_dir + "/checkpoints/Checkpoint_Example.pth")
            self.elevonly_vw2instab_model = _load_model(elevonly_vw2instab_model_checkpoint, model_type='Travformer', device=self.device)
            print("Travformer model loaded.")
        else:
            print("Skipping Travformer model loading (read-only mode).")







    @torch.no_grad()
    def TravFormer_Inference(self, target_instability, elevation_map_numpy, commands_numpy, model, device='cpu'):

        cmdvorcmdw = 0 if commands_numpy[0,0] != 0 else 1

        # Preprocess inputs
        elevation_map = torch.tensor(elevation_map_numpy.copy(), dtype=torch.float32, requires_grad=True).unsqueeze(1).to(device)
        commands = torch.tensor(commands_numpy.copy(), dtype=torch.float32, requires_grad=True).to(device)


        pred, pred_logstd = model(elevation_map, commands)
        pred_logstd = torch.exp(pred_logstd)

        if target_instability is not None:
            mean_grad_commands = torch.autograd.grad(pred, commands, grad_outputs=torch.ones_like(pred), retain_graph=True, create_graph=True)[0]
            std_grad_commands = torch.autograd.grad(pred_logstd, commands, grad_outputs=torch.ones_like(pred_logstd), create_graph=True, retain_graph=True, allow_unused=True)[0]

            pred = pred.cpu().detach().numpy()
            pred_logstd = pred_logstd.cpu().detach().numpy()
            mean_grad_commands = mean_grad_commands.cpu().detach().numpy()
            std_grad_commands = std_grad_commands.cpu().detach().numpy()

            residual = (target_instability - ((pred[:,0]) + instability_std_multiplier*np.exp((pred_logstd[:,0])) ) )
            total_gradient = (mean_grad_commands[:,cmdvorcmdw]) + instability_std_multiplier*(std_grad_commands[:,cmdvorcmdw]) * np.exp((pred_logstd[:,0])) 
            delta_vorw = total_gradient**(-1) * residual
            target_vorw = (commands_numpy[:,cmdvorcmdw]) + delta_vorw

        else:
            pred = pred.cpu().detach().numpy()
            pred_logstd = pred_logstd.cpu().detach().numpy()

        return pred, pred_logstd












    def Initilize_map(self):

        if self.is_RobocentricMap_received:

        # Initialize global map
            if not self.is_ElevationMap_built:
                self.is_ElevationMap_built = True

            self.map_size_rows = int(self.map_size_x/self.map_resolution)
            self.map_size_cols = int(self.map_size_y/self.map_resolution)

            self.ElevationMap = np.full((self.map_size_rows, self.map_size_cols), np.nan)

        # Build global trav map  (flag not initialized here though)
        # 3D (x,y,theta(=heading angle)) global map that contains the pre-estimated cmd v and w array
            self.TraversabilityMap = np.full((self.map_size_rows, self.map_size_cols, self.TraversabilityMap_size_layers, 4), np.nan)
            self.TraversabilityMap[:,:,:,0] = self.InitialGuess_cmd_v
            self.TraversabilityMap[:,:,:,1] = self.InitialGuess_cmd_w
            self.TraversabilityMap[:,:,:,2] = np.nan
            self.TraversabilityMap[:,:,:,3] = np.nan

        # Build metainfo map
            self.ElevMap_MetaInfo = np.full((self.map_size_rows, self.map_size_cols), ElevmapInfo_IntEnum.ELEV_NotInitialized.value)
            self.TravMap_MetaInfo = np.full((self.map_size_rows, self.map_size_cols, self.TravMap_MetaInfo_size_layers), TravmapInfo_IntEnum.TRAV_NotInitialized.value)

        


    

        


    def Update_map(self, path_plan = None, visualize_map = False):
        
        """
        # In grid map, the left top corner ( = (0,0) ) is the max x and y point in the map, regardless of the robot pose.
        # increasing row/col indexing results in decreasing x/y in world coordinate.
        # x increases as row decreases, y increases as col decreases
        # This rule is the same both for local and global map
        """


        if not self.is_ElevationMap_built:
            self.Initilize_map() # it goes to backend Initilize_map if backend is running

        if self.is_RobocentricMap_received and self.is_ElevationMap_built:

            # Snapshot the callback-owned data atomically so the entire Update_map
            # iteration uses a consistent (position, layers) pair from the same message.
            with self._map_data_lock:
                layer_data_snap = dict(self.layer_data)
                center_pos_snap = self.RobocentricMap_CenterPosOffset_Worldfr

            # Update global map
            # rospy.loginfo("Available layers in self.layer_data: %s", list(layer_data_snap.keys()))

            if len(self.which_layer) == 1:
                local_map = layer_data_snap[self.which_layer[0]]
                nan_map = local_map
            else:

                if self.mean_over_maps == True:
                    local_map = np.mean([layer_data_snap[self.which_layer[i]] for i in range(len(self.which_layer))], axis=0)
                    nan_map = layer_data_snap[self.which_layer[1]]
                else:
                    local_map = layer_data_snap[self.which_layer[0]]
                    nan_map = layer_data_snap[self.which_layer[1]]
                # the result will be NaN if one of the layers is NaN

            # local_map  : (H, W) float  – robocentric elevation grid received this tick
            # nan_map    : (H, W) float  – layer used to identify unobserved cells (NaN = no data)
            H, W = local_map.shape
            # half_r / half_c: pixel offset from the local map's top-left corner to its center cell,
            # used to convert (row, col) indices into world-frame metric offsets
            half_r = int(self.RobocentricMap_Rows / 2)
            half_c = int(self.RobocentricMap_Cols / 2)
            R         = self.TravUpdate_XYResolution  # coarse grid resolution (m) used to quantise TravUpdate candidate positions
            half_patch = 0.5 * self.local_patch_size  # half the Travformer inference window (m); candidates closer to the map edge than this are skipped
            NINIT_T = self.TravmapInfo_IntEnum.TRAV_NotInitialized.value  # int sentinel: traversability not yet estimated for a direction
            NINIT_E = self.ElevmapInfo_IntEnum.ELEV_NotInitialized.value  # int sentinel: elevation cell never written

            # ---------------------------------------------------------------
            # Step 1: Map every local cell to its world-frame position and
            #         corresponding index in the global elevation map.
            #
            # Convention (shared by local and global maps):
            #   row 0 = highest x; row increases → x decreases
            #   col 0 = highest y; col increases → y decreases
            # ---------------------------------------------------------------

            # row_idx / col_idx : (H, W) int – full index grids for the local map
            row_idx, col_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

            # global_x_all / global_y_all : (H, W) float – world-frame metric position of each local cell
            global_x_all = (half_r - row_idx) * self.map_resolution + center_pos_snap[0]
            global_y_all = (half_c - col_idx) * self.map_resolution + center_pos_snap[1]

            # row_global_all / col_global_all : (H, W) int – corresponding row/col in the global elevation map
            row_global_all = ((self.map_xmax - global_x_all) / self.map_resolution).astype(int)
            col_global_all = ((self.map_ymax - global_y_all) / self.map_resolution).astype(int)

            # ---------------------------------------------------------------
            # Step 2: Build a validity mask and flatten to per-valid-cell vectors.
            #
            # A cell is valid if it has sensor data (non-NaN) AND its world
            # position falls within the pre-allocated global map extent.
            # Cells outside the global map are silently dropped (the map does
            # not auto-expand).
            # ---------------------------------------------------------------

            # valid : (H, W) bool – True for cells that have data and lie inside the global map
            valid = (~np.isnan(nan_map)
                     & (row_global_all >= 0) & (row_global_all < self.map_size_rows)
                     & (col_global_all >= 0) & (col_global_all < self.map_size_cols))

            # All vectors below are shape (N,) where N = number of valid cells this tick

            vrl = row_idx[valid]       # local map row indices
            vcl = col_idx[valid]       # local map col indices
            vrg = row_global_all[valid]  # global map row indices
            vcg = col_global_all[valid]  # global map col indices
            vgx = global_x_all[valid]  # world-frame x (m)
            vgy = global_y_all[valid]  # world-frame y (m)
            local_vals = local_map[vrl, vcl]              # (N,) float – incoming elevation values
            elev_meta  = self.ElevMap_MetaInfo[vrg, vcg]  # (N,) int   – current ElevmapInfo_IntEnum status per cell

            # ---------------------------------------------------------------
            # Step 3: Classify each valid cell into one of three triggering
            #         conditions that require queueing nearby grid points for
            #         traversability re-estimation.
            #
            #   Case A – Cell newly seen: global cell had no data yet.
            #            Queue the coarse-grid neighbourhood so the Travformer
            #            can estimate traversability for this region.
            #
            #   Case B – Terrain changed: elevation differs from the stored
            #            value by more than elevation_diff_threshold.
            #            The traversability estimate is stale and must be
            #            recomputed regardless of current TravMap state.
            #
            #   Case C – Gap-filling probe: the global cell is already
            #            measured, but some traversability directions are still
            #            unestimated (e.g. the wavefront hasn't reached here
            #            yet). A random subsample of such cells is queued to
            #            gradually fill these gaps.
            #
            # Cases are mutually exclusive in priority order A > B > C.
            # ---------------------------------------------------------------

            # case_a : (N,) bool – cells seen for the first time
            case_a = elev_meta == NINIT_E

            # elev_diff : (N,) float – |new elevation − stored elevation| per cell
            elev_diff = np.abs(local_vals - self.ElevationMap[vrg, vcg])
            # case_b : (N,) bool – cells whose terrain changed significantly (and are not under reconstruction)
            case_b = (~case_a
                      & (elev_diff > self.elevation_diff_threshold)
                      & (elev_meta <= ElevmapInfo_IntEnum.ELEV_MEASURED.value))

            # case_c : (N,) bool – stochastic gap-filling; computed only for cells not already handled
            # by A/B to match the original per-cell RNG consumption pattern
            can_be_c = ~case_a & ~case_b  # (N,) bool – cells eligible for Case C
            case_c = np.zeros(len(vgx), dtype=bool)
            if np.any(can_be_c):
                # trav_any_uninit : (K,) bool – at least one heading direction has no traversability estimate
                #   K = number of can_be_c cells; TravMap_MetaInfo[r,c] has shape (num_layers,) per cell
                trav_any_uninit = np.any(self.TravMap_MetaInfo[vrg[can_be_c], vcg[can_be_c]] == NINIT_T, axis=-1)
                # rand_pass : (K,) bool – random gate so only ~RandomRate_TravUpdate fraction are probed
                rand_pass = np.random.random(int(np.sum(can_be_c))) < self.RandomRate_TravUpdate
                case_c[can_be_c] = trav_any_uninit & rand_pass

            # ---------------------------------------------------------------
            # Step 4: Collect traversability update candidate positions into
            #         self.TravUpdate_Pts.
            #
            # TravUpdate_Pts : (M, 2) float – world-frame (x, y) positions of
            #   coarse-grid points that need Travformer inference. M grows as
            #   new terrain is discovered and shrinks after inference runs.
            #
            # Candidates are quantised to the coarse TravUpdate_XYResolution
            # grid so the Travformer is invoked at well-spaced locations, not
            # once per fine-resolution elevation cell.
            #
            # Cases A and B use the floor-aligned 2×2 neighbourhood of the
            # triggering cell's coarse-grid square (4 candidate positions).
            # Case C uses the single nearest rounded grid point and only adds
            # it if at least one neighbouring grid point is already estimated
            # (so the estimation wavefront expands inward, not outward).
            # ---------------------------------------------------------------

            # existing_set: Python set of (x, y) float64 tuples mirroring TravUpdate_Pts for O(1) dedup
            existing_set = set(map(tuple, self.TravUpdate_Pts)) if len(self.TravUpdate_Pts) > 0 else set()
            new_pts = []  # accumulates new (2,) arrays before a single vstack at the end

            def _floor_to_cands(gx, gy):
                # gx, gy : (K,) float – world coords of triggering cells
                # Returns (4*M, 2) float – 4 floor-grid neighbours for each of M unique floor points.
                # np.arange(fx-R, fx+R, R) gives [fx-R, fx], so offsets span {-R, 0} × {-R, 0}.
                fx = R * np.floor(gx / R)
                fy = R * np.floor(gy / R)
                uniq = np.unique(np.stack([fx, fy], axis=1), axis=0)  # (M, 2) – deduplicated floor points
                offsets = np.array([[-R, -R], [-R, 0.], [0., -R], [0., 0.]])  # (4, 2)
                return (uniq[:, None, :] + offsets[None]).reshape(-1, 2)      # (4M, 2)

            def _base_filter(pts):
                # pts : (P, 2) float – candidate world positions
                # Removes candidates outside the global map and within half_patch of any map boundary
                # (the Travformer needs a full patch around each inference point).
                # Returns (pts_filtered, rg, cg) each of length ≤ P.
                rg = ((self.map_xmax - pts[:, 0]) / self.map_resolution).astype(int)  # (P,) global row
                cg = ((self.map_ymax - pts[:, 1]) / self.map_resolution).astype(int)  # (P,) global col
                keep = ((rg >= 0) & (rg < self.map_size_rows)
                        & (cg >= 0) & (cg < self.map_size_cols)
                        & (np.abs(self.map_xmax - pts[:, 0]) >= half_patch)
                        & (np.abs(pts[:, 0] - self.map_xmin) >= half_patch)
                        & (np.abs(self.map_ymax - pts[:, 1]) >= half_patch)
                        & (np.abs(pts[:, 1] - self.map_ymin) >= half_patch))
                return pts[keep], rg[keep], cg[keep]

            def _add_new(pts):
                # pts : (Q, 2) float – pre-filtered candidates to add
                # Skips any point already in TravUpdate_Pts (via set lookup) and appends the rest.
                for pt in pts:
                    t = tuple(pt)
                    if t not in existing_set:
                        existing_set.add(t)
                        new_pts.append(pt)

            # --- Case A: newly discovered cells ---
            # Queue the coarse-grid neighbourhood only if every traversability direction at
            # that candidate is still uninitialized (any already-estimated neighbour is left alone).
            if np.any(case_a):
                cands, crg, ccg = _base_filter(_floor_to_cands(vgx[case_a], vgy[case_a]))
                if len(cands) > 0:
                    # all_uninit : (Q,) bool – True if NO direction at this candidate has been estimated yet
                    all_uninit = np.all(self.TravMap_MetaInfo[crg, ccg] == NINIT_T, axis=-1)
                    _add_new(cands[all_uninit])

            # --- Case B: terrain changed ---
            # Re-queue unconditionally — the existing traversability estimate is stale.
            if np.any(case_b):
                cands, crg, ccg = _base_filter(_floor_to_cands(vgx[case_b], vgy[case_b]))
                _add_new(cands)

            # --- Case C: gap-filling wavefront ---
            # Add a rounded candidate only if it neighbours already-estimated terrain (wavefront
            # expansion inward) and still has at least one unestimated direction.
            if np.any(case_c):
                # rounded_x/y : (K,) float – cell's world position snapped to the nearest coarse-grid point
                rounded_x = R * np.round(vgx[case_c] / R)
                rounded_y = R * np.round(vgy[case_c] / R)
                # uniq_c : (M, 2) float – deduplicated rounded candidate positions
                uniq_c = np.unique(np.stack([rounded_x, rounded_y], axis=1), axis=0)
                cands, crg, ccg = _base_filter(uniq_c)  # (Q, 2), (Q,), (Q,)

                if len(cands) > 0:
                    # Build the 4 floor-aligned neighbours for each candidate to check whether
                    # the estimation wavefront has already reached this vicinity.
                    offsets = np.array([[-R, -R], [-R, 0.], [0., -R], [0., 0.]])  # (4, 2)
                    nbr     = cands[:, None, :] + offsets[None]  # (Q, 4, 2) – 4 neighbours per candidate
                    nbr_rg  = ((self.map_xmax - nbr[:, :, 0]) / self.map_resolution).astype(int)  # (Q, 4)
                    nbr_cg  = ((self.map_ymax - nbr[:, :, 1]) / self.map_resolution).astype(int)  # (Q, 4)
                    # nbr_valid : (Q, 4) bool – which neighbours fall inside the global map
                    nbr_valid = ((nbr_rg >= 0) & (nbr_rg < self.map_size_rows)
                                 & (nbr_cg >= 0) & (nbr_cg < self.map_size_cols))

                    # near_preest : (Q,) bool – True if any in-bounds neighbour has at least one
                    # estimated traversability direction (i.e. the wavefront has reached nearby)
                    near_preest = np.zeros(len(cands), dtype=bool)
                    for j in range(4):
                        v = nbr_valid[:, j]  # (Q,) bool – which candidates have a valid j-th neighbour
                        if not np.any(v):
                            continue
                        # Clamp out-of-bounds indices to 0 for safe array access; masked out by `v` below
                        rg_j = np.where(v, nbr_rg[:, j], 0)
                        cg_j = np.where(v, nbr_cg[:, j], 0)
                        # has_nonnull : (Q,) bool – neighbour j is valid AND has some estimated direction
                        has_nonnull = v & np.any(self.TravMap_MetaInfo[rg_j, cg_j] != NINIT_T, axis=-1)
                        near_preest |= has_nonnull

                    # Discard candidates not adjacent to the estimation wavefront
                    cands = cands[near_preest]
                    crg   = crg[near_preest]
                    ccg   = ccg[near_preest]

                    if len(cands) > 0:
                        # any_uninit : (Q,) bool – candidate still has at least one unestimated direction
                        any_uninit = np.any(self.TravMap_MetaInfo[crg, ccg] == NINIT_T, axis=-1)
                        _add_new(cands[any_uninit])

            # Commit all new candidates in one allocation instead of one np.append per point
            if new_pts:
                self.TravUpdate_Pts = np.vstack([self.TravUpdate_Pts, np.array(new_pts)])

            # ---------------------------------------------------------------
            # Step 5: Write the new elevation values into the global map.
            # ---------------------------------------------------------------

            self.ElevationMap[vrg, vcg] = local_vals
            # newly_init : (N,) bool – cells written for the very first time; promote their status
            newly_init = elev_meta == NINIT_E
            self.ElevMap_MetaInfo[vrg[newly_init], vcg[newly_init]] = ElevmapInfo_IntEnum.ELEV_MEASURED.value




            # ---------------------------------------------------------------
            # Step 6: Prune TravUpdate_Pts of stale entries.
            #
            # Remove any queued candidate whose elevation is NaN in the
            # (now-updated) global map — Travformer inference requires valid
            # elevation data.
            # ---------------------------------------------------------------

            deleting_indices = []
            for i in range(self.TravUpdate_Pts.shape[0]):

                estimating_gridpt = self.TravUpdate_Pts[i]  # (2,) float – world (x, y) of this candidate
                estimating_pt_global_row = int( (self.map_xmax - estimating_gridpt[0])/self.map_resolution )
                estimating_pt_global_col = int( (self.map_ymax - estimating_gridpt[1])/self.map_resolution )

                if self.is_TraversabilityMap_built and np.isnan(self.ElevationMap[estimating_pt_global_row, estimating_pt_global_col]):
                    # This pt is nan. no need to preestimate. turning off only when this is the first time to build preest map
                    deleting_indices.append(i)

            self.TravUpdate_Pts = np.delete(self.TravUpdate_Pts, deleting_indices, axis=0)





            if visualize_map:
                # Update the traversability map
                print("\nUpdating traversability map......")
                start_time = time()
                print(" Current robot position: {:.1f}, {:.1f}, {:.0f}".format(self.robot_x, self.robot_y, np.rad2deg(self.robot_heading)))
                print(" Global map range:      x-axis: ", self.map_xmin, self.map_xmax, "y-axis: ", self.map_ymin, self.map_ymax)
                print(" Estimating", self.TravUpdate_Pts.shape[0], " points in total...")
                

            self.get_travmap(self.TravUpdate_Pts, 0, 0, 0, 0)

            if visualize_map:
                print("Traversability map updated. Total time of estimation: ", time()-start_time, "\n")

            theta_to_goal = np.arctan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
            theta_robot = Utils.wrap_to_pi(self.robot_heading)
            if visualize_map and self.is_TraversabilityMap_built:
                self.visualize_maps(theta_robot, path_plan)


    def visualize_maps(self, map_theta, path_plan = None):
        

        theta_layer = int(np.round((map_theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
        if theta_layer == self.TraversabilityMap_size_layers:
            theta_layer = 0 # pi = -pi


        path_plan = np.array(path_plan).reshape(-1,2) if path_plan is not None else None
        if path_plan is not None and path_plan.shape[0] > 0:
            path_plan_InGridMap = np.zeros_like(path_plan)
            for i in range(path_plan.shape[0]):
                path_plan_InGridMap[i,:] = self.xy2grid(path_plan[i,0], path_plan[i,1])




        # ####################################### Elevation map #######################################
        # Only visualize elevation map if it exists
        if self.ElevationMap is not None:
            global savefigureonce_elevmap
            if not hasattr(self, '_fig1') or not plt.fignum_exists(1):
                self._fig1 = plt.figure(1)
                manager = self._fig1.canvas.manager
                manager.window.setGeometry(0, 0, 480, 420)  # Adjust the position and size as needed
            else:
                self._fig1.clf()
            plt.figure(1)
            plt.rcParams.update(RC_PARAMS)
            plt.imshow(self.ElevationMap[:,:], cmap='viridis')
            plt.title("Global Elevation Map", fontproperties=times_new_roman, fontsize=FS_TITLE)
            if path_plan is not None and path_plan.shape[0] > 0:
                plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], "Dr", markersize=3) 
                plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], '-r', linewidth=1.5)
                plt.plot(path_plan_InGridMap[0,1], path_plan_InGridMap[0,0], marker="D",color=(0,1,1), markersize=6)
                plt.plot(path_plan_InGridMap[-1,1], path_plan_InGridMap[-1,0], marker="*",color=(1,1,0), markersize=10)
            
            cbar = plt.colorbar()
            cbar.set_label('Height (m)', fontsize=FS_LEGEND, fontproperties=times_new_roman)  # Addi
            plt.xlabel("Y-axis", fontsize=FS_LABEL, fontproperties=times_new_roman)
            plt.ylabel("X-axis", fontsize=FS_LABEL, fontproperties=times_new_roman)
            plt.xticks(np.linspace(0, self.ElevationMap.shape[1], int(self.map_ymax-self.map_ymin)+1), np.linspace(self.map_ymax, self.map_ymin, int(self.map_ymax-self.map_ymin+1) ))
            plt.yticks(np.linspace(0, self.ElevationMap.shape[0], int(self.map_xmax-self.map_xmin)+1), np.linspace(self.map_xmax, self.map_xmin, int(self.map_xmax-self.map_xmin+1) ))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            # plt.minorticks_on()
            plt.tick_params(axis='both', which='major', labelsize=FS_TICK, width=2)
            for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
                label.set_fontproperties(times_new_roman)
                label.set_fontsize(FS_TICK)

            if savefigureonce_elevmap and path_plan is not None and path_plan.shape[0] > 0:
                figure_path = os.path.join(Path(__file__).parents[1], "./")
                os.makedirs(figure_path, exist_ok=True)
                plt.savefig(figure_path + 'World Elevation Map' + '.pdf', format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
                savefigureonce_elevmap = False
            plt.show(block=False)
            plt.pause(0.001)      






        ####################################### Traversability map #######################################
        global savefigureonce_travmap
        if not hasattr(self, '_fig2') or not plt.fignum_exists(2):
            self._fig2 = plt.figure(2)
            manager = self._fig2.canvas.manager
            manager.window.setGeometry(0, 660, 480, 420)   # Adjust the position and size as needed
            # manager.window.setGeometry(0, 0, 480, 420)
        else:
            self._fig2.clf()
        plt.figure(2)
        plt.rcParams.update(RC_PARAMS)


        plt.imshow(self.TraversabilityMap[:,:,theta_layer,self.viz_channel], cmap='coolwarm_r', vmin=self.viz_vmin, vmax=self.viz_vmax)
        plt.title(f"Traversability Map", fontproperties=times_new_roman, fontsize=FS_TITLE)
        cbar = plt.colorbar()
        cbar.set_label(self.viz_cbar_label, fontproperties=times_new_roman, fontsize=FS_LEGEND)  #
        

        if path_plan is not None and path_plan.shape[0] > 0:
            # plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], "Dr", markersize=3) 
            plt.plot([x[1] for x in path_plan_InGridMap], [x[0] for x in path_plan_InGridMap], '--', color='lime', linewidth=4, label = 'Global Path Plan')
            # plt.plot(path_plan_InGridMap[0,1], path_plan_InGridMap[0,0], marker="D",color=(0,1,1), markersize=6)
            # plt.plot(path_plan_InGridMap[-1,1], path_plan_InGridMap[-1,0], marker="*",color=(1,1,0), markersize=10)
            pass

        plt.xlabel("Y-axis", fontproperties=times_new_roman, fontsize=FS_LABEL)
        plt.ylabel("X-axis", fontproperties=times_new_roman, fontsize=FS_LABEL)
        plt.xticks(np.linspace(0, self.TraversabilityMap.shape[1], int(self.map_ymax-self.map_ymin+1)), np.linspace(self.map_ymax, self.map_ymin, int(self.map_ymax-self.map_ymin+1)))
        plt.yticks(np.linspace(0, self.TraversabilityMap.shape[0], int(self.map_xmax-self.map_xmin+1)), np.linspace(self.map_xmax, self.map_xmin, int(self.map_xmax-self.map_xmin+1)))
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=FS_TICK, width=2)
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontproperties(times_new_roman)
            label.set_fontsize(FS_TICK)
        legend = plt.legend(loc='upper left', prop=times_new_roman, fontsize=FS_LEGEND)
        for text in legend.get_texts():
            text.set_fontsize(FS_LEGEND)  # 🔹 Force larger size

        if savefigureonce_travmap and path_plan is not None and path_plan.shape[0] > 0:
            figure_path = os.path.join(Path(__file__).parents[1], "./")
            os.makedirs(figure_path, exist_ok=True)
            plt.savefig(figure_path + 'Traversability Map' + '.pdf', format=PLOT_FORMAT, dpi=PLOT_DPI, bbox_inches='tight')
            savefigureonce_travmap = False
        plt.show(block=False)
        plt.pause(0.001)


    def get_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        self._compute_travmap(estimating_xypts_set, xmin, xmax, ymin, ymax)

        if not self.is_TraversabilityMap_built:
            self.is_TraversabilityMap_built = True
            if hasattr(self, 'backend'):
                self.backend.set_trav_map_built(True)

        self.TravUpdate_Pts = np.empty((0, 2))

    def _compute_travmap(self, estimating_xypts_set, xmin, xmax, ymin, ymax):
        raise NotImplementedError("Subclasses must implement _compute_travmap")






                



    
    def _compute_patch_indices(self, x_center, y_center, yaw, width, height):
        """Compute the map row/col indices for a yaw-aligned patch centered at (x_center, y_center).
        SHOULD BE APPLIED ONLY FOR FUll SIZE MAP

        width: local x-axis extent (meters)
        height: local y-axis extent (meters)
        Returns rows, cols, valid, (half_width_grid, half_height_grid).

        N = 2 * (half_width_grid) * (half_height_grid)
        rows: 2D numpy array of shape (N, 1) with the row indices of the patch
        cols: 2D numpy array of shape (N, 1) with the column indices of the patch
        valid: 1D boolean array of shape (N,) indicating whether each index is valid (in the map)
        (half_width_grid, half_height_grid): tuple of half the patch size in grid cells
        """
        row_center = int((self.map_xmax - x_center) / self.map_resolution)
        col_center = int((self.map_ymax - y_center) / self.map_resolution)
        givenpoint_tf_grid = tf3.affines.compose(
            np.array([row_center, col_center, 0]),
            tf3.euler.euler2mat(0, 0, yaw, 'sxyz'),
            np.ones(3)
        )
        half_width_grid  = np.round(0.5 * width  / self.map_resolution).astype(int)
        half_height_grid = np.round(0.5 * height / self.map_resolution).astype(int)
        local_patch_index = np.array([
            [i, j, 0]
            for i in range(-half_width_grid, half_width_grid)
            for j in range(-half_height_grid, half_height_grid)
        ])
        local_patch_index_homogeneous = np.hstack(
            (local_patch_index, np.ones((local_patch_index.shape[0], 1)))
        )
        local_patch_index_transformed = np.round(
            np.dot(givenpoint_tf_grid, local_patch_index_homogeneous.T).T[:, :2]
        ).astype(int)

        rows = local_patch_index_transformed[:, 0]
        cols = local_patch_index_transformed[:, 1]
        valid = (rows >= 0) & (rows < self.map_size_rows) & \
                (cols >= 0) & (cols < self.map_size_cols)

        return rows, cols, valid, (half_width_grid, half_height_grid)

    def get_patch_in_map(self, x_center, y_center, yaw, width, height, map):
        """
        Get a local patch of the global elevation map, centered at (x_center, y_center). yaw-aligned direction.

        SHOULD BE APPLIED ONLY FOR FUll SIZE MAP

        x_center, y_center: in the world frame
        width: local x-axis extent in meters
        height: local y-axis extent in meters
        yaw: in radian
        map: numpy array with at least 2 dimensions (H, W, ...)

        Returns:
        local_patch: array of shape (half_width_grid*2, half_height_grid*2, ...) matching extra map dimensions
        ismapnan: a boolean indicating whether the local patch contains any nan value
        """
        ismapnan = False

        if map is not None:
            rows, cols, valid, (half_width_grid, half_height_grid) = \
                self._compute_patch_indices(x_center, y_center, yaw, width, height)

            safe_rows = np.where(valid, rows, 0)
            safe_cols = np.where(valid, cols, 0)
            gathered = map[safe_rows, safe_cols]  # shape: (N,) or (N, ...) for extra dims
            extra_dims = gathered.shape[1:]
            valid_broadcast = valid.reshape(-1, *([1] * len(extra_dims)))
            nan_fill = np.full(extra_dims, np.nan, dtype=gathered.dtype)
            values = np.where(valid_broadcast, gathered, nan_fill)

            ismapnan = bool(np.any(np.isnan(values)))
            local_patch = values.reshape((half_width_grid * 2, half_height_grid * 2) + extra_dims)

            return local_patch, ismapnan

    def set_patch_in_map(self, x_center, y_center, yaw, width, height, map, patch):
        """
        Write a local patch into the global map at (x_center, y_center), yaw-aligned.
        patch: 2D (or higher-D) numpy array or scalar value.
        SHOULD BE APPLIED ONLY FOR FUll SIZE MAP

        Uses an inverse-transform approach so that every global map cell whose
        centre falls inside the rotated patch rectangle is written.  Cells that
        the forward-transform would have missed (rotation-induced holes) are
        filled by bilinear interpolation (float patches) or nearest-neighbour
        (integer patches).
        """
        if map is None or patch is None:
            return

        row_center = int((self.map_xmax - x_center) / self.map_resolution)
        col_center = int((self.map_ymax - y_center) / self.map_resolution)
        hw = int(np.round(0.5 * width  / self.map_resolution))
        hh = int(np.round(0.5 * height / self.map_resolution))

        # Rotation matrix (same z-axis convention as _compute_patch_indices)
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        R     = np.array([[cos_yaw, -sin_yaw],
                          [sin_yaw,  cos_yaw]])
        R_inv = R.T   # [[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]]

        # Axis-aligned bounding box of the rotated patch in global map space
        corners = np.array([[-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh]], dtype=float)
        corners_g = (R @ corners.T).T + [row_center, col_center]

        row_min = max(int(np.floor(corners_g[:, 0].min())), 0)
        row_max = min(int(np.ceil( corners_g[:, 0].max())), self.map_size_rows - 1)
        col_min = max(int(np.floor(corners_g[:, 1].min())), 0)
        col_max = min(int(np.ceil( corners_g[:, 1].max())), self.map_size_cols - 1)

        if row_min > row_max or col_min > col_max:
            return

        # Dense grid of every global cell in the bounding box
        gr, gc = np.meshgrid(
            np.arange(row_min, row_max + 1),
            np.arange(col_min, col_max + 1),
            indexing='ij'
        )

        # Inverse-rotate to local patch coordinates
        drow = (gr - row_center).astype(float)
        dcol = (gc - col_center).astype(float)
        local_i = R_inv[0, 0] * drow + R_inv[0, 1] * dcol
        local_j = R_inv[1, 0] * drow + R_inv[1, 1] * dcol

        # Keep only cells whose centre lies inside the patch region
        # (half-open interval matches the original range(-hw, hw))
        inside = (local_i >= -hw) & (local_i < hw) & \
                 (local_j >= -hh) & (local_j < hh)

        if not isinstance(patch, np.ndarray):
            map[gr[inside], gc[inside]] = patch
            return

        if patch.shape[0] != hw * 2 or patch.shape[1] != hh * 2:
            raise ValueError(
                f"Patch shape {patch.shape[:2]} does not match expected ({hw * 2}, {hh * 2})."
            )

        # Continuous patch indices (0-based) for interpolation
        pi = local_i[inside] + hw   # in [0, 2*hw)
        pj = local_j[inside] + hh   # in [0, 2*hh)
        coords = np.array([pi, pj])

        # Bilinear for floats, nearest-neighbour for integers
        interp_order = 0 if np.issubdtype(patch.dtype, np.integer) else 1
        extra_dims = patch.shape[2:]

        def _interp_slice(slc):
            vals = map_coordinates(slc.astype(float), coords,
                                   order=interp_order, mode='nearest')
            if np.issubdtype(patch.dtype, np.integer):
                vals = np.round(vals).astype(patch.dtype)
            return vals

        if len(extra_dims) == 0:
            map[gr[inside], gc[inside]] = _interp_slice(patch)
        else:
            out = np.empty((int(inside.sum()),) + extra_dims, dtype=patch.dtype)
            for idx in np.ndindex(extra_dims):
                out[(slice(None),) + idx] = _interp_slice(patch[(Ellipsis,) + idx])
            map[gr[inside], gc[inside]] = out





    def get_theta_layer(self, theta):
        theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
        if theta_layer == self.TraversabilityMap_size_layers:
            theta_layer = 0 # pi = -pi
        return theta_layer

    def xy2grid(self, x, y):
        
        row = int( (self.map_xmax - x)/self.map_resolution )
        col = int( (self.map_ymax - y)/self.map_resolution )
        
        if row < 0 or row > self.map_size_rows:
            print(f"Out of range. x: {x}, y: {y}, row: {row}, col: {col}")
            return None, None
        elif row == self.map_size_rows:
            row = self.map_size_rows - 1

        if col < 0 or col > self.map_size_cols:
            print(f"Out of range. x: {x}, y: {y}, row: {row}, col: {col}")
            return None, None
        elif col == self.map_size_cols:
            col = self.map_size_cols - 1

        return row, col

    def grid2xy(self, row, col):
        x = self.map_xmax - row * self.map_resolution
        y = self.map_ymax - col * self.map_resolution
        return x, y




    def get_cmd_limits(self):

        # TODO: there could be latency issue for robot's position when called by MPC in the future.


        if self.map_resolution is None or self.ElevationMap is None or self.TraversabilityMap is None:
            return self.toorisky_v, self.toorisky_w
        
        cmd_v_limit = self.maximum_cmd_v
        cmd_w_limit = self.maximum_cmd_w


        discretization = safecmd_sampling_dist
        num_points = int(lookahead_distance/discretization)
        angle_tolerance = np.deg2rad(lookahead_angle)

        counter = 0
        for i in range(num_points):
            delta_dist = i * discretization
            
            for j in range(-1, 2):
                theta = self.robot_heading + j * angle_tolerance
                theta = Utils.wrap_to_pi(theta)

                x = self.robot_x + delta_dist * np.cos(theta)
                y = self.robot_y + delta_dist * np.sin(theta)

                row = int( (self.map_xmax - x)/self.map_resolution )
                col = int( (self.map_ymax - y)/self.map_resolution )
                

                theta_layer = int(np.round((theta-self.TraversabilityMap_theta_min)/self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0 # pi = -pi

                if row < 0 or row >= self.map_size_rows or col < 0 or col >= self.map_size_cols:
                    continue

                cmd_v = self.TraversabilityMap[row, col, theta_layer, 0]
                cmd_w = self.TraversabilityMap[row, col, theta_layer, 1]


                cmd_v_limit = ( cmd_v_limit * counter + cmd_v )/(counter + 1)
                cmd_w_limit = ( cmd_w_limit * counter + cmd_w )/(counter + 1)
                counter += 1

                # if cmd_v != 0:
                #     cmd_v_limit = np.min([cmd_v_limit, cmd_v])
                #     cmd_w_limit = np.min([cmd_w_limit, cmd_w])
                # else:
                #     print("You are near obstacles")
        

        return cmd_v_limit, cmd_w_limit


    def get_waypoint(self, robot_x, robot_y, robot_heading, total_lookahead_time, cmd_v_limit, cmd_w_limit, path):
        """
        This function takes in the current position of the robot and returns the next waypoint (in the global coordinate system)
        that the robot should move to. The waypoint is selected from the global path such that the robot is expected to reach
        the waypoint within total_lookahead_time seconds.

        The function first finds the node in the global path that is nearest to the robot. It then iterates through the path
        from this node and finds the node that the robot is expected to reach within the given total time. The function then
        returns the coordinates of this node as the next waypoint.

        If the robot is close to the end of the path, the function will return the last node in the path as the next waypoint.
        If the robot is close to a node in the path, the function will return the next node in the path as the next waypoint.

        The function also takes into account the elevation map and the current orientation of the robot. The function will
        return a waypoint that is reachable by the robot given the elevation map and the current orientation of the robot.

        Parameters
        ----------
        robot_x : float
            The current x-coordinate of the robot in the global coordinate system.
        robot_y : float
            The current y-coordinate of the robot in the global coordinate system.
        robot_heading : float
            The current heading of the robot in radians.
        total_lookahead_time : float
            Total lookahead time in seconds (= waypoint_lookahead_time from config).
        cmd_v_limit : float
            Maximum linear velocity limit.
        cmd_w_limit : float
            Maximum angular velocity limit.
        path : array_like
            The global path as a numpy array of shape (N, 2) where each row is [x, y].

        Returns
        -------
        global_waypoint : array_like or False
            The coordinates of the next waypoint in the global coordinate system, or False if no path is available.
        """
        path = np.array(path)
        global_waypoint = False
        total_time = total_lookahead_time
        if path.shape[0] == 0:
            print(' Getting Waypoint: No path received')
            return False
        
        cmd_v_limit = max(cmd_v_limit, 0.1)
        cmd_w_limit = max(cmd_w_limit, 0.06)

        # find a node that the agent can reach within the total time
        node_index_nearest_from_robot = int(np.argmin([math.hypot(path[nd,0] - robot_x, path[nd,1] - robot_y)
                                        for nd in range(path.shape[0])]))
        waypoint_node_index = node_index_nearest_from_robot
        robot_to_wp_angle = np.arctan2( (path[waypoint_node_index,1]-robot_y), (path[waypoint_node_index,0]-robot_x)  )
        robot_to_goal_angle = np.arctan2( (path[-1,1]-robot_y), (path[-1,0]-robot_x)  )
        robot_to_goal_distance = math.hypot(path[-1,0] - robot_x, path[-1,1] - robot_y)
        reachable_radius = total_time * cmd_v_limit

        if waypoint_node_index >= path.shape[0] - 1:
            print(' Getting Waypoint: Getting the end of the path')
            return path[-1,:]

        # Find an intermediate goal that is distant by 2 * reachable_radius from the robot
        min_cmd = max(cmd_v_limit, 0.25)
        intermediate_goal_radius = 2 * total_time * min_cmd
        intermediate_goal_node_index = waypoint_node_index
        for node_idx in range(waypoint_node_index,path.shape[0]):
            if math.hypot(path[node_idx,0] - robot_x, path[node_idx,1] - robot_y) >= intermediate_goal_radius:
                intermediate_goal_node_index = node_idx
                break
        robot_to_intermediate_goal_distance = math.hypot(path[intermediate_goal_node_index,0] - robot_x, path[intermediate_goal_node_index,1] - robot_y)
        
        set_this_node_as_waypoint = False
        while set_this_node_as_waypoint == False:

            wp_to_intermediate_goal_distance = math.hypot(path[waypoint_node_index,0] - path[intermediate_goal_node_index,0], path[waypoint_node_index,1] - path[intermediate_goal_node_index,1])

            if waypoint_node_index >= path.shape[0] - 1:
                print(' Getting Waypoint: Getting the end of the path')
                return path[-1,:]

            if wp_to_intermediate_goal_distance > robot_to_intermediate_goal_distance:
                # This waypoint is behind the robot. Proceed to the next node
                waypoint_node_index = waypoint_node_index + 1
                robot_to_wp_angle = np.arctan2( (path[waypoint_node_index,1]-robot_y), (path[waypoint_node_index,0]-robot_x)  )

            else:
                set_this_node_as_waypoint = True

        if waypoint_node_index >= path.shape[0] - 1:
            print(' Getting Waypoint: Getting the end of the path')
            return path[-1,:]

        if self.is_TraversabilityMap_built:
            
            # Starting from the initial waypoint, find a point that the robot can reach within the total time
            for node in range(waypoint_node_index, path.shape[0]):

                robot_to_this_node_distance = math.hypot(path[node,0] - robot_x, path[node,1] - robot_y)
                if robot_to_this_node_distance > reachable_radius: # we got to a node that is further than the reachable radius

                    if node == waypoint_node_index: # the nearest hopeful node is already further than the reachable radius
                        global_waypoint = np.array([robot_x, robot_y]) + (path[node,:] - np.array([robot_x, robot_y])) * (reachable_radius)/robot_to_this_node_distance
                        return global_waypoint

                    if node == path.shape[0] - 1:
                        global_waypoint = path[-1,:]
                        return global_waypoint

                    # This means that the found node is not the nearest hopeful node.
                    # We need to get intermediate point between this found node and its previous node
                    previous_node = node - 1
                    robot_to_previous_node_distance = math.hypot(path[previous_node,0] - robot_x, path[previous_node,1] - robot_y)
                    if robot_to_previous_node_distance > reachable_radius:
                        print(f'[DEBUG INFO] The previous node: {path[previous_node,:]} with distance {robot_to_previous_node_distance} is further than the reachable radius: {reachable_radius}')
                        print(f'[DEBUG INFO] The found node: {path[node,:]} with distance {robot_to_this_node_distance} is further than the reachable radius: {reachable_radius}')
                        print('But it is not the nearest hopeful node')
                        raise ValueError('This should not happen')
                    else: # robot_to_previous_node_distance < reachable_radius
                        global_waypoint = path[previous_node,:] + (path[node,:] - path[previous_node,:]) * (reachable_radius - robot_to_previous_node_distance)/(robot_to_this_node_distance - robot_to_previous_node_distance)
                        # This is not exactly accurate equation but good approximation
                        return global_waypoint

        return global_waypoint

    
    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tf_trans.euler_from_quaternion(quat)
        msg_yaw = yaw
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_roll = Utils.wrap_to_pi(roll)
        self.robot_pitch = Utils.wrap_to_pi(pitch)
        self.robot_heading = Utils.wrap_to_pi(yaw)
        # rospy.loginfo_throttle(1.0, "[pose_callback] x: %.2f, y: %.2f, yaw: %.2f rad", self.robot_x, self.robot_y, self.robot_heading)


    def robo_centric_map_callback(self, msg):

        if not self.is_RobocentricMap_received:
            self.is_RobocentricMap_received = True
        # start = time()
        # rospy.loginfo("Received GridMap message")

        # rospy.loginfo("Header:\n%s" % msg.info.header)

        # Get the dimensions of the map
        self.map_resolution = msg.info.resolution
        self.RobocentricMap_LengthY = msg.info.length_y
        self.RobocentricMap_LengthX = msg.info.length_x

        self.RobocentricMap_Rows = int(self.RobocentricMap_LengthY / self.map_resolution)
        self.RobocentricMap_Cols = int(self.RobocentricMap_LengthX / self.map_resolution)
        # rospy.loginfo("Map dimensions: %dx%d" % (self.RobocentricMap_Rows, self.RobocentricMap_Cols))

        # Build layer arrays before acquiring the lock to minimize lock hold time
        layers = msg.layers
        new_center_pos = (msg.info.pose.position.x, msg.info.pose.position.y, msg.info.pose.position.z)
        new_center_rot = (msg.info.pose.orientation.x, msg.info.pose.orientation.y, msg.info.pose.orientation.z, msg.info.pose.orientation.w)
        new_layer_data = {}
        for i, layer in enumerate(layers):
            if not layer in self.which_layer:
                continue
            data_layer = np.array(msg.data[i].data)
            new_layer_data[layer] = data_layer.reshape((self.RobocentricMap_Rows, self.RobocentricMap_Cols)).transpose() #Don't know why but transpose is needed. row indexing

        # Atomically publish the new position and layer data so Update_map always
        # sees a consistent pair (same message's position + same message's data).
        with self._map_data_lock:
            self.RobocentricMap_CenterPosOffset_Worldfr = new_center_pos
            self.RobocentricMap_CenterRotOffset_Worldfr = new_center_rot
            self.layer_data.update(new_layer_data)
        # rospy.loginfo("Map position: %s" % str(self.RobocentricMap_CenterPosOffset_Worldfr))
        # rospy.loginfo("Layers: %s" % layers)

        # rospy.loginfo("Received GridMap message in %f seconds" % (time() - start))



























                                        








class CMDbasedMap(BaseMap):
    def __init__(self, env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_Travformer = True):
        super().__init__(env_xmin, env_xmax, env_ymin, env_ymax, goal_x, goal_y, which_layer, preest_update_resolution, instab_limit, load_Travformer)


        self.InitialGuess_cmd_v = cfg.default_safe_cmd_v
        self.InitialGuess_cmd_w = cfg.default_safe_cmd_w

        self.InitialGuess_auxiliary_score = np.nan
        self.InitialGuess_auxiliary_score_std = np.nan


        # trav as velcity-based method specific
        self.toorisky_v = cfg.toorisky_v
        self.toorisky_w = cfg.toorisky_w
        self.v_res = self.maximum_cmd_v / cfg.num_commands
        self.w_res = self.maximum_cmd_w / cfg.num_commands
        # TreatObs_cmd_v: minimum speed bin considered safe; patches failing at this speed keep toorisky_v sentinel
        self.TreatObs_cmd_v = cfg.TreatObs_cmd_v
        if self.TreatObs_cmd_v > self.maximum_cmd_v:
            raise ValueError(
                f"TreatObs_cmd_v ({self.TreatObs_cmd_v}) must be <= maximum_cmd_v ({self.maximum_cmd_v}). "
                "No speed bins would exist and every cell would be marked toorisky."
            )

        self.instability_limit = instab_limit # = \delta_limit
        self.interp_sigmoid_k = 14.0  # sharpness of S-curve interpolation between quadrant corners

        # Visualization
        self.viz_channel = 0  # default channel
        self.viz_vmin = 0.0
        self.viz_vmax = 2.0
        self.viz_title = "Traversability Map"
        self.viz_cbar_label = "Linear Velocity (m/s)"


    def _compute_travmap(self, estimating_xypts_set, estimating_area_xmin, estimating_area_xmax, estimating_area_ymin, estimating_area_ymax):
        
        if estimating_xypts_set is None:
            print(" Estimating area: x-axis: {:.1f}, {:.1f}, y-axis: {:.1f}, {:.1f}".format(estimating_area_xmin, estimating_area_xmax, estimating_area_ymin, estimating_area_ymax))
                    
            if estimating_area_xmin < self.map_xmin or estimating_area_xmax > self.map_xmax or estimating_area_ymin < self.map_ymin or estimating_area_ymax > self.map_ymax:
                print("The estimating area is out of the global map.")


            # Grid points in the estimating area, grid size of self.TravUpdate_XYResolution. xyz coordinates
            estimating_xycpts_set = np.array([ [x,y,c] for x in np.arange(estimating_area_xmin, estimating_area_xmax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution) for y in np.arange(estimating_area_ymin, estimating_area_ymax + self.TravUpdate_XYResolution, self.TravUpdate_XYResolution)\
                                                for c in np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution) ])
        
        else:
            # estimating_xypts_set is ndarray of [x,y]. I want to add one more dimension for theta
            _thetas = np.arange(self.TraversabilityMap_theta_min, self.TraversabilityMap_theta_max + self.TraversabilityMap_theta_resolution, self.TraversabilityMap_theta_resolution)
            estimating_xycpts_set = np.column_stack([
                np.repeat(estimating_xypts_set, len(_thetas), axis=0),
                np.tile(_thetas, len(estimating_xypts_set))
            ])






        if self.is_RobocentricMap_received and self.is_ElevationMap_built and estimating_xycpts_set.shape[0] != 0:

            # ── Step 1: Unpack query points ──────────────────────────────────────────
            # N: total number of (x, y, theta) points whose traversability will be estimated.
            N      = estimating_xycpts_set.shape[0]
            xs     = estimating_xycpts_set[:, 0]   # world x,   shape (N,)
            ys     = estimating_xycpts_set[:, 1]   # world y,   shape (N,)
            thetas = estimating_xycpts_set[:, 2]   # heading,   shape (N,)

            # ── Step 2: Patch geometry ────────────────────────────────────────────────
            # Each query point gets a square elevation patch of side P (in pixels) centered on it.
            # half_w: half-side in pixels; P: full side length in pixels.
            half_w = int(np.round(0.5 * self.local_patch_size / self.map_resolution))
            P = 2 * half_w

            # local_i, local_j: per-pixel row/col offsets from patch center in body frame, shape (P*P,).
            # Built once and shared across all N points.
            ii, jj = np.meshgrid(range(-half_w, half_w), range(-half_w, half_w), indexing='ij')
            local_i = ii.ravel()
            local_j = jj.ravel()

            # ── Step 3: Body-frame → map-pixel transform ──────────────────────────────
            # Map convention: row = (xmax - world_x) / res, col = (ymax - world_y) / res.
            # row_centers, col_centers: map-pixel of each query point center, shape (N,).
            row_centers = ((self.map_xmax - xs) / self.map_resolution).astype(int)
            col_centers = ((self.map_ymax - ys) / self.map_resolution).astype(int)
            cos_t = np.cos(thetas)   # shape (N,)
            sin_t = np.sin(thetas)   # shape (N,)

            # Rotate body-frame pixel offsets by theta, then shift to map-frame center.
            # global_rows[n,k] = cos(θ_n)*local_i[k] - sin(θ_n)*local_j[k] + row_center_n
            # global_cols[n,k] = sin(θ_n)*local_i[k] + cos(θ_n)*local_j[k] + col_center_n
            # Shape: (N, P*P)
            global_rows = np.round(
                cos_t[:, None] * local_i - sin_t[:, None] * local_j + row_centers[:, None]
            ).astype(int)
            global_cols = np.round(
                sin_t[:, None] * local_i + cos_t[:, None] * local_j + col_centers[:, None]
            ).astype(int)

            # ── Step 4: Elevation lookup with bounds check ────────────────────────────
            # valid: True where the rotated pixel falls inside the elevation map, shape (N, P*P).
            valid = (
                (global_rows >= 0) & (global_rows < self.map_size_rows) &
                (global_cols >= 0) & (global_cols < self.map_size_cols)
            )
            # Clamp OOB indices to 0 so array lookup never faults, then replace those values with NaN.
            safe_rows = np.where(valid, global_rows, 0)
            safe_cols = np.where(valid, global_cols, 0)
            gathered = self.ElevationMap[safe_rows, safe_cols]   # (N, P*P)
            gathered = np.where(valid, gathered, np.nan)

            # A patch is unusable if ANY pixel is NaN (OOB or missing elevation data).
            ismapnan_arr = np.any(np.isnan(gathered), axis=1)    # (N,) bool

            all_patches = gathered.reshape(N, P, P)              # (N, P, P)

            # ── Step 5: Build xyc_patchindex ─────────────────────────────────────────
            # xyc_patchindex: (N, 4) — columns [x, y, theta, patch_index].
            # patch_index = -1  → patch was NaN / unusable (will use InitialGuess later).
            # patch_index = k   → patch is the k-th row in patch_tensor (0-indexed, 0 ≤ k < M).
            xyc_patchindex = np.concatenate(
                (estimating_xycpts_set, np.full((N, 1), -1, dtype=int)), axis=1
            )
            valid_mask = ~ismapnan_arr       # (N,) — True for fully-observed patches
            M = int(valid_mask.sum())        # number of usable patches

            if M > 0:
                valid_patches = all_patches[valid_mask]                # (M, P, P)
                # Height normalization: subtract each patch's 5th-percentile elevation so the
                # model sees relative relief rather than absolute height.
                offsets = np.percentile(valid_patches.reshape(M, -1), 5, axis=1)  # (M,)
                valid_patches = valid_patches - offsets[:, None, None]
                patch_tensor = valid_patches                           # (M, P, P)
                # Assign sequential indices 0..M-1 to valid rows so they can look up target_v/w later.
                xyc_patchindex[valid_mask, 3] = np.arange(M)
            else:
                patch_tensor = np.empty((0, P, P))

            # ── Step 6: Batch padding and duplication ─────────────────────────────────
            # The model requires batch sizes that are multiples of 64.
            # '% 64' on the outer expression prevents adding 64 spurious zeros when M is already a multiple of 64.
            pad = (64 - patch_tensor.shape[0] % 64) % 64
            patch_tensor_extended = np.concatenate(
                (patch_tensor, np.zeros((pad, P, P))), axis=0
            )  # (size, P, P), size = next multiple of 64 ≥ M
            # Duplicate: first 'size' rows → cmd_v evaluation, next 'size' rows → cmd_w evaluation.
            patch_cat_tensor = np.concatenate((patch_tensor_extended, patch_tensor_extended), axis=0)  # (2*size, P, P)
            size = patch_tensor_extended.shape[0]  # multiple of 64; only indices 0..M-1 hold real patches

            # ── Step 7: Candidate speed bins ──────────────────────────────────────────
            # target_v/w: (size, 1) — best safe speed per patch slot.
            # Initialized to sentinel values (toorisky) meaning "no safe speed found yet".
            target_v = np.full((size, 1), self.toorisky_v)
            v_res = self.v_res
            # v_bins: candidate linear speeds in descending order [v_max, ..., TreatObs_cmd_v].
            # Lower bound is TreatObs_cmd_v: patches failing at every bin keep toorisky_v (treated as obstacle).
            # Descending so the largest feasible speed is assigned first and locked in by the sentinel guard.
            v_bins = np.arange(self.TreatObs_cmd_v, self.maximum_cmd_v + v_res, v_res)
            v_bins = v_bins[::-1]

            target_w = np.full((size, 1), self.toorisky_w)
            w_res = self.w_res
            # w_bins: candidate angular speeds in descending order [w_max, ..., w_res].
            w_bins = np.arange(w_res, self.maximum_cmd_w + w_res, w_res)
            w_bins = w_bins[::-1]

            # ── Step 8: Chunked TravFormer inference ──────────────────────────────────
            # CMD_CHUNK: number of velocity bins batched into a single GPU forward pass.
            # Higher = fewer kernel launches (faster), but scales GPU memory by CMD_CHUNK.
            # Halved automatically on CUDA OOM until it reaches 1 (original per-bin behavior).
            # Skip inference entirely when size==0 (all patches were NaN/OOB): every
            # xyc_patchindex entry already has patchindex==-1, so Step 9 will fall back
            # to InitialGuess for all points without needing TravFormer output.
            CMD_CHUNK = 4
            n_bins = len(v_bins)  # total number of candidate speed levels to evaluate
            chunk_start = 0
            while size > 0 and chunk_start < n_bins:
                # chunk_v/chunk_w: (n_chunk,) — velocity bins in this chunk (descending order,
                # so the largest feasible speed is assigned first and later bins are skipped by the mask guard)
                chunk_v = v_bins[chunk_start : chunk_start + CMD_CHUNK]
                chunk_w = w_bins[chunk_start : chunk_start + CMD_CHUNK]
                n_chunk = len(chunk_v)  # ≤ CMD_CHUNK; may be smaller for the last chunk

                # patch_chunk: (n_chunk*2*size, P, P)
                # patch_cat_tensor is (2*size, P, P): first half for cmd_v eval, second for cmd_w eval.
                # Replicated n_chunk times so that each bin j occupies rows [j*2*size : (j+1)*2*size].
                patch_chunk = np.concatenate([patch_cat_tensor] * n_chunk, axis=0)

                # cmd_chunk: (n_chunk*2*size, 2) — command pairs [cmd_v, cmd_w].
                # For bin j:
                #   rows [j*2*size       : j*2*size+size] → col 0 = v, col 1 = 0  (v-only evaluation)
                #   rows [j*2*size+size  : (j+1)*2*size ] → col 0 = 0, col 1 = w  (w-only evaluation)
                cmd_chunk = np.zeros((n_chunk * 2 * size, 2))
                for j, (v, w) in enumerate(zip(chunk_v, chunk_w)):
                    cmd_chunk[j*2*size        : j*2*size + size,  0] = v
                    cmd_chunk[j*2*size + size : (j+1)*2*size,     1] = w

                try:
                    start = time()
                    # instab_mean_cat, instab_std_cat: each (n_chunk*2*size, 1)
                    [instab_mean_cat, instab_std_cat] = self.TravFormer_Inference(None, patch_chunk, cmd_chunk, model=self.elevonly_vw2instab_model, device=self.device)
                except RuntimeError as e:
                    err = str(e).lower()
                    # Catch both OOM and "invalid configuration argument" (CUDA grid-dim overflow
                    # when batch×num_heads exceeds the 65535 hardware limit at large resolutions).
                    if ('out of memory' in err or 'invalid configuration argument' in err) and CMD_CHUNK > 1:
                        CMD_CHUNK = max(1, CMD_CHUNK // 2)
                        del patch_chunk, cmd_chunk  # free CPU memory before retry
                        if self.device != 'cpu':
                            torch.cuda.empty_cache()
                        continue  # retry this chunk with halved size
                    raise

                # VaR_chunk: (n_chunk, 2*size) — Value-at-Risk instability per (bin, patch).
                # VaR_chunk[j, :size]  → v-evaluation VaR for bin j across all patches
                # VaR_chunk[j, size:]  → w-evaluation VaR for bin j across all patches
                VaR_chunk = (instab_mean_cat[:,0] + instability_std_multiplier*instab_std_cat[:,0]).reshape(n_chunk, 2*size)

                for j, (v, w) in enumerate(zip(chunk_v, chunk_w)):
                    VaR = VaR_chunk[j]  # (2*size,)
                    # Assign v only to patches still at sentinel (not yet assigned a safe speed)
                    # that now pass the VaR threshold at this speed level.
                    mask_v = (VaR[:size] < self.instability_limit) & (target_v[:, 0] == self.toorisky_v)
                    target_v[mask_v, 0] = v
                    mask_w = (VaR[size:] < self.instability_limit) & (target_w[:, 0] == self.toorisky_w)
                    target_w[mask_w, 0] = w

                chunk_start += n_chunk

                # Early exit: all patches have been assigned a safe speed for both v and w
                if not (target_v[:, 0] == self.toorisky_v).any() and not (target_w[:, 0] == self.toorisky_w).any():
                    break
            
            # Temporarily over-ride target_w Command 
            safe = target_v[:, 0] != self.toorisky_v
            target_w[safe, 0] = self.maximum_cmd_w * target_v[safe, 0] / self.maximum_cmd_v

            



            
            # ── Step 9: Write safe speeds back to TraversabilityMap ──────────────────
            # Iterate all N query points in the same order as estimating_xycpts_set so that
            # patchindex_counter stays aligned with xyc_patchindex rows.
            patchindex_counter = 0
            for xyc in estimating_xycpts_set:

                x     = xyc[0]
                y     = xyc[1]
                theta = xyc[2]

                # patchindex: index into target_v/w for this point.
                # -1 means the patch was NaN → fall back to the initial-guess safe speed.
                patchindex = xyc_patchindex[patchindex_counter, 3].astype(int)
                patchindex_counter += 1

                if patchindex == -1:
                    cmd_v = self.InitialGuess_cmd_v
                    cmd_w = self.InitialGuess_cmd_w
                else:
                    # target_v/w[patchindex] holds the largest speed that passed the VaR threshold,
                    # or toorisky_v/w (sentinel) if no speed was safe for this patch.
                    cmd_v = target_v[patchindex, 0]
                    cmd_w = target_w[patchindex, 0]

                # The TraversabilityMap is updated in grid cells of size TravUpdate_XYResolution.
                # Compute the world-space corners of the square cell centered at (x, y):
                #   BL = bottom-left (smaller x, smaller y)
                #   TR = top-right   (larger  x, larger  y)
                square_BL_xyc = np.array([x - self.TravUpdate_XYResolution/2, y - self.TravUpdate_XYResolution/2])
                square_TR_xyc = np.array([x + self.TravUpdate_XYResolution/2, y + self.TravUpdate_XYResolution/2])

                # Convert world corners to map pixel indices.
                # Map convention: row = (xmax - world_x)/res  → larger x = smaller row (TR → smaller row).
                #                 col = (ymax - world_y)/res  → larger y = smaller col (TR → smaller col).
                # So row_start < row_end and col_start < col_end, giving a valid slice.
                square_BL_entry = np.array([ round( (self.map_xmax - square_BL_xyc[0])/self.map_resolution ), round( (self.map_ymax - square_BL_xyc[1])/self.map_resolution ) ])
                square_TR_entry = np.array([ round( (self.map_xmax - square_TR_xyc[0])/self.map_resolution ), round( (self.map_ymax - square_TR_xyc[1])/self.map_resolution ) ])

                # Discretize heading to the nearest TraversabilityMap layer.
                # wrap_to_pi ensures theta ∈ (-π, π]; the modulo wrap handles θ=π ≡ θ=-π.
                theta = Utils.wrap_to_pi(theta)
                theta_layer = int(np.round((theta - self.TraversabilityMap_theta_min) / self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0  # π and -π map to the same layer

                # Clip pixel bounds to stay within the map and write cmd_v, cmd_w, and status.
                row_start = int(np.clip(square_TR_entry[0], 0, self.map_size_rows))
                row_end   = int(np.clip(square_BL_entry[0], 0, self.map_size_rows))
                col_start = int(np.clip(square_TR_entry[1], 0, self.map_size_cols))
                col_end   = int(np.clip(square_BL_entry[1], 0, self.map_size_cols))
                self.TraversabilityMap[row_start:row_end, col_start:col_end, theta_layer, 0] = cmd_v
                self.TraversabilityMap[row_start:row_end, col_start:col_end, theta_layer, 1] = cmd_w
                self.TravMap_MetaInfo[row_start:row_end, col_start:col_end, theta_layer] = self.TravmapInfo_IntEnum.TRAV_Estimated.value










            # Interpolation
            N = math.floor(self.TravUpdate_XYResolution / self.map_resolution)
            R = self.TravUpdate_XYResolution
            _t   = np.linspace(0.0, 1.0, N + 1)
            _sig = 1.0 / (1.0 + np.exp(-self.interp_sigmoid_k * (_t - 0.5)))
            xi_w = (_sig - _sig[0]) / (_sig[-1] - _sig[0])    # normalize to [0,1]; f(0.5)=0.5 exactly
            yi_w = xi_w

            # (dxTL, dyTL, dxBR, dyBR, dxTR, dyTR) world-space offsets from BL corner
            # quadrants 0-3 are defined in array space (same as original)
            _q_offsets = [
                ( R,  0,  0, -R,  R, -R),  # q0
                ( 0,  R,  R,  0,  R,  R),  # q1
                (-R,  0,  0,  R, -R,  R),  # q2
                ( 0, -R, -R,  0, -R, -R),  # q3
            ]

            for xyc in estimating_xycpts_set:
                theta = Utils.wrap_to_pi(xyc[2])
                theta_layer = math.floor(np.round((theta - self.TraversabilityMap_theta_min) / self.TraversabilityMap_theta_resolution))
                if theta_layer == self.TraversabilityMap_size_layers:
                    theta_layer = 0  # pi = -pi

                row_BL, col_BL = self.xy2grid(xyc[0], xyc[1])
                if row_BL is None or col_BL is None:
                    continue

                for quadrant, (dxTL, dyTL, dxBR, dyBR, dxTR, dyTR) in enumerate(_q_offsets):
                    row_TL, col_TL = self.xy2grid(xyc[0] + dxTL, xyc[1] + dyTL)
                    if row_TL is None or col_TL is None:
                        continue
                    row_BR, col_BR = self.xy2grid(xyc[0] + dxBR, xyc[1] + dyBR)
                    if row_BR is None or col_BR is None:
                        continue
                    row_TR, col_TR = self.xy2grid(xyc[0] + dxTR, xyc[1] + dyTR)
                    if row_TR is None or col_TR is None:
                        continue

                    # Corner values for both map layers at once: shape (2,)
                    BL = self.TraversabilityMap[row_BL, col_BL, theta_layer, :]
                    TL = self.TraversabilityMap[row_TL, col_TL, theta_layer, :]
                    BR = self.TraversabilityMap[row_BR, col_BR, theta_layer, :]
                    TR = self.TraversabilityMap[row_TR, col_TR, theta_layer, :]

                    # Bilinear interpolation over the (N+1)x(N+1) patch, both layers at once
                    bottom = BL + xi_w[:, None] * (BR - BL)                         # (N+1, 2)
                    top    = TL + xi_w[:, None] * (TR - TL)                         # (N+1, 2)
                    patch  = (1 - yi_w[:, None, None]) * bottom[None] \
                           +      yi_w[:, None, None]  * top[None]                  # (N+1, N+1, 2)

                    # Write patch into the map; orientation per quadrant derived from loop indices:
                    #   q0: TravMap[row_BL-yi, col_BL+xi]  →  patch[::-1]
                    #   q1: TravMap[row_BL-xi, col_BL-yi]  →  patch[::-1,::-1].transpose(1,0,2)
                    #   q2: TravMap[row_BL+yi, col_BL-xi]  →  patch[:,    ::-1]
                    #   q3: TravMap[row_BL+xi, col_BL+yi]  →  patch.transpose(1,0,2)
                    if quadrant == 0:
                        _interp_vals = patch[::-1]
                        _row_sl, _col_sl = slice(row_BL-N, row_BL+1),  slice(col_BL,   col_BL+N+1)
                    elif quadrant == 1:
                        _interp_vals = patch[::-1, ::-1].transpose(1, 0, 2)
                        _row_sl, _col_sl = slice(row_BL-N, row_BL+1),  slice(col_BL-N, col_BL+1)
                    elif quadrant == 2:
                        _interp_vals = patch[:, ::-1]
                        _row_sl, _col_sl = slice(row_BL,   row_BL+N+1), slice(col_BL-N, col_BL+1)
                    elif quadrant == 3:
                        _interp_vals = patch.transpose(1, 0, 2)
                        _row_sl, _col_sl = slice(row_BL,   row_BL+N+1), slice(col_BL,   col_BL+N+1)
                    self.TraversabilityMap[_row_sl, _col_sl, theta_layer, :] = _interp_vals
