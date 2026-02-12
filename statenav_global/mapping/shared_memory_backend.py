"""
Shared Memory Backend - Clean Separation of Concerns

This module provides a backend abstraction for shared memory that can be used
by CMDbasedMap without modifying its methods. The backend handles all shared
memory operations, and CMDbasedMap just uses arrays normally.

Design Philosophy:
- Shared memory logic is completely separate from map logic
- CMDbasedMap methods work unchanged - they just use arrays from backend
- Backend can be swapped (shared memory vs local memory) transparently
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Tuple, Optional, Dict
import time
import ctypes
import atexit
import signal
import sys


class SharedMemoryBackend:
    """
    Backend that manages shared memory for map arrays.
    
    This class handles:
    - Creating/attaching to shared memory
    - Providing numpy array views
    - Synchronization (locks, version numbers)
    - Metadata storage
    
    CMDbasedMap uses this backend instead of creating arrays directly.
    """
    
    def __init__(self, mode='writer', shared_metadata=None):
        """
        Initialize backend
        
        Args:
            mode: 'writer' (creates shared memory) or 'reader' (attaches to existing)
            shared_metadata: Optional SharedMetadata object from parent process.
                           If None, creates new metadata (for separate processes).
                           If provided, uses shared metadata (for parent-spawned processes).
        """
        self.mode = mode
        self._initialized = False
        
        # Shared memory blocks
        self.shm_blocks = {}
        
        # Array views (these point to shared memory)
        self.arrays = {}
        
        # Metadata (stored in shared memory for coordination)
        if shared_metadata is not None:
            # Use shared metadata from parent (parent-spawning approach)
            self.metadata = shared_metadata
        else:
            # Create new metadata (separate processes approach)
            self.metadata = self._create_metadata()
        
        # Register cleanup handlers for crash recovery
        if mode == 'writer':
            atexit.register(self._cleanup_on_exit)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            # Note: SIGKILL cannot be caught, but stale memory will be cleaned on next startup
        
    def _create_metadata(self):
        """
        Create shared metadata structure
        
        Note: This creates NEW metadata. For parent-spawning approach,
        pass shared_metadata from parent to __init__ instead.
        """
        class SharedMetadata:
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
                
                # Locks
                self.write_lock = mp.Lock()  # Exclusive: Only ONE writer at a time
                self.read_lock = mp.RLock()  # Shared: Multiple readers can acquire simultaneously
                
        return SharedMetadata()
    
    def initialize_arrays(self,
                         elev_map_rows: int,
                         elev_map_cols: int,
                         trav_map_rows: int,
                         trav_map_cols: int,
                         trav_map_layers: int,
                         map_resolution: float,
                         elev_map_xmin: float,
                         elev_map_xmax: float,
                         elev_map_ymin: float,
                         elev_map_ymax: float):
        """
        Initialize shared memory arrays
        
        Args:
            Map dimensions and parameters
        """
        if self._initialized:
            raise RuntimeError("Backend already initialized")
        
        # Store metadata
        self.metadata.map_resolution.value = map_resolution
        self.metadata.elev_map_rows.value = elev_map_rows
        self.metadata.elev_map_cols.value = elev_map_cols
        self.metadata.trav_map_rows.value = trav_map_rows
        self.metadata.trav_map_cols.value = trav_map_cols
        self.metadata.trav_map_layers.value = trav_map_layers
        self.metadata.elev_map_xmin.value = elev_map_xmin
        self.metadata.elev_map_xmax.value = elev_map_xmax
        self.metadata.elev_map_ymin.value = elev_map_ymin
        self.metadata.elev_map_ymax.value = elev_map_ymax
        
        # Calculate sizes
        elev_map_size = elev_map_rows * elev_map_cols * np.dtype(np.float32).itemsize
        trav_map_size = trav_map_rows * trav_map_cols * trav_map_layers * 4 * np.dtype(np.float32).itemsize
        elev_meta_size = elev_map_rows * elev_map_cols * np.dtype(np.int32).itemsize
        trav_meta_size = trav_map_rows * trav_map_cols * trav_map_layers * np.dtype(np.int32).itemsize
        
        if self.mode == 'writer':
            self._create_shared_memory(elev_map_size, trav_map_size, elev_meta_size, trav_meta_size)
        else:
            self._attach_to_shared_memory()
        
        # Create numpy array views
        self._create_array_views(elev_map_rows, elev_map_cols, trav_map_rows, 
                                 trav_map_cols, trav_map_layers)
        
        self._initialized = True
    
    def _create_shared_memory(self, elev_size, trav_size, elev_meta_size, trav_meta_size):
        """Create new shared memory blocks (writer mode)"""
        try:
            self.shm_blocks['trav_map'] = shared_memory.SharedMemory(
                create=True, size=trav_size, name='trav_map_shm'
            )
            self.shm_blocks['elev_map'] = shared_memory.SharedMemory(
                create=True, size=elev_size, name='elev_map_shm'
            )
            self.shm_blocks['elev_meta'] = shared_memory.SharedMemory(
                create=True, size=elev_meta_size, name='elev_meta_shm'
            )
            self.shm_blocks['trav_meta'] = shared_memory.SharedMemory(
                create=True, size=trav_meta_size, name='trav_meta_shm'
            )
            print(f"Created shared memory: TravMap={trav_size/1e6:.1f}MB, ElevMap={elev_size/1e6:.1f}MB")
        except FileExistsError:
            # Stale shared memory from previous crash - clean it up automatically
            print("Warning: Stale shared memory detected from previous crash. Auto-cleaning...")
            self._cleanup_stale_memory()
            # Retry creation
            self.shm_blocks['trav_map'] = shared_memory.SharedMemory(
                create=True, size=trav_size, name='trav_map_shm'
            )
            self.shm_blocks['elev_map'] = shared_memory.SharedMemory(
                create=True, size=elev_size, name='elev_map_shm'
            )
            self.shm_blocks['elev_meta'] = shared_memory.SharedMemory(
                create=True, size=elev_meta_size, name='elev_meta_shm'
            )
            self.shm_blocks['trav_meta'] = shared_memory.SharedMemory(
                create=True, size=trav_meta_size, name='trav_meta_shm'
            )
            print("Auto-cleaned stale memory and recreated shared memory")
    
    def _cleanup_stale_memory(self):
        """Clean up stale shared memory from previous crash"""
        stale_names = ['trav_map_shm', 'elev_map_shm', 'elev_meta_shm', 'trav_meta_shm']
        for name in stale_names:
            try:
                stale_shm = shared_memory.SharedMemory(name=name)
                stale_shm.close()
                stale_shm.unlink()
            except FileNotFoundError:
                pass  # Already cleaned up
            except Exception:
                pass  # Ignore errors
    
    def _attach_to_shared_memory(self, timeout: float = 10.0):
        """Attach to existing shared memory (reader mode)"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.shm_blocks['trav_map'] = shared_memory.SharedMemory(name='trav_map_shm')
                self.shm_blocks['elev_map'] = shared_memory.SharedMemory(name='elev_map_shm')
                self.shm_blocks['elev_meta'] = shared_memory.SharedMemory(name='elev_meta_shm')
                self.shm_blocks['trav_meta'] = shared_memory.SharedMemory(name='trav_meta_shm')
                print("Attached to existing shared memory")
                return
            except FileNotFoundError:
                time.sleep(0.1)
        raise RuntimeError(f"Could not attach to shared memory within {timeout} seconds")
    
    def _create_array_views(self, elev_rows, elev_cols, trav_rows, trav_cols, trav_layers):
        """Create numpy array views that point to shared memory"""
        self.arrays['TraversabilityMap'] = np.ndarray(
            (trav_rows, trav_cols, trav_layers, 4),
            dtype=np.float32,
            buffer=self.shm_blocks['trav_map'].buf
        )
        
        self.arrays['ElevationMap'] = np.ndarray(
            (elev_rows, elev_cols),
            dtype=np.float32,
            buffer=self.shm_blocks['elev_map'].buf
        )
        
        self.arrays['ElevMap_MetaInfo'] = np.ndarray(
            (elev_rows, elev_cols),
            dtype=np.int32,
            buffer=self.shm_blocks['elev_meta'].buf
        )
        
        self.arrays['TravMap_MetaInfo'] = np.ndarray(
            (trav_rows, trav_cols, trav_layers),
            dtype=np.int32,
            buffer=self.shm_blocks['trav_meta'].buf
        )
        
        # Initialize arrays (only for writer)
        if self.mode == 'writer':
            self.arrays['TraversabilityMap'].fill(np.nan)
            self.arrays['ElevationMap'].fill(np.nan)
            self.arrays['ElevMap_MetaInfo'].fill(0)
            self.arrays['TravMap_MetaInfo'].fill(0)
    
    # ========================================================================
    # Public API: Array Access
    # ========================================================================
    
    def get_traversability_map(self) -> np.ndarray:
        """Get TraversabilityMap array (reads/writes to shared memory)"""
        return self.arrays['TraversabilityMap']
    
    def get_elevation_map(self) -> np.ndarray:
        """Get ElevationMap array (reads/writes to shared memory)"""
        return self.arrays['ElevationMap']
    
    def get_elev_meta_info(self) -> np.ndarray:
        """Get ElevMap_MetaInfo array (reads/writes to shared memory)"""
        return self.arrays['ElevMap_MetaInfo']
    
    def get_trav_meta_info(self) -> np.ndarray:
        """Get TravMap_MetaInfo array (reads/writes to shared memory)"""
        return self.arrays['TravMap_MetaInfo']
    
    # ========================================================================
    # Public API: Synchronization
    # ========================================================================
    
    def acquire_write_lock(self):
        """
        Acquire write lock (REQUIRED before updating maps)
        
        Write lock is EXCLUSIVE - only ONE writer can hold it at a time.
        This prevents:
        - Multiple writers updating simultaneously (race conditions)
        - Corruption from concurrent writes
        
        Note: Readers can still read while writer has write lock (individual
        element reads are atomic), but writer blocks other writers.
        """
        self.metadata.write_lock.acquire()
    
    def release_write_lock(self):
        """Release write lock (call after updating maps)"""
        self.metadata.write_lock.release()
    
    def acquire_read_lock(self):
        """
        Acquire read lock (OPTIONAL - for consistency guarantees)
        
        Read lock is SHARED - multiple readers can hold it simultaneously.
        
        When to use:
        - If you need to ensure consistency across MULTIPLE reads
        - If you're reading a large region and want snapshot consistency
        - If you want to prevent reading during bulk updates
        
        When NOT needed:
        - Individual element reads are atomic (float32/int32)
        - Single element access: trav_map[row, col, layer, 0] is safe without lock
        - Most planning operations (they read individual elements)
        
        Example:
            # Without lock (fine for individual elements):
            value = trav_map[10, 20, 0, 0]  # ✅ Atomic, safe
            
            # With lock (if you need consistency across multiple reads):
            acquire_read_lock()
            val1 = trav_map[10, 20, 0, 0]
            val2 = trav_map[11, 21, 0, 0]  # Ensures both from same "snapshot"
            release_read_lock()
        """
        self.metadata.read_lock.acquire()
    
    def release_read_lock(self):
        """Release read lock"""
        self.metadata.read_lock.release()
    
    def increment_version(self):
        """Increment version number (call after map updates)"""
        with self.metadata.version.get_lock():
            self.metadata.version.value += 1
    
    def get_version(self) -> int:
        """Get current version number"""
        return self.metadata.version.value
    
    # ========================================================================
    # Public API: Metadata Access
    # ========================================================================
    
    def set_trav_map_built(self, value: bool):
        """Set traversability map built flag"""
        with self.metadata.is_trav_map_built.get_lock():
            self.metadata.is_trav_map_built.value = value
    
    def set_elev_map_built(self, value: bool):
        """Set elevation map built flag"""
        with self.metadata.is_elev_map_built.get_lock():
            self.metadata.is_elev_map_built.value = value
    
    def get_trav_map_built(self) -> bool:
        """Get traversability map built flag"""
        return self.metadata.is_trav_map_built.value
    
    def get_elev_map_built(self) -> bool:
        """Get elevation map built flag"""
        return self.metadata.is_elev_map_built.value
    
    def get_map_resolution(self) -> float:
        """Get map resolution"""
        return self.metadata.map_resolution.value
    
    def get_map_bounds(self) -> Tuple[float, float, float, float]:
        """Get map bounds (xmin, xmax, ymin, ymax)"""
        return (self.metadata.elev_map_xmin.value,
                self.metadata.elev_map_xmax.value,
                self.metadata.elev_map_ymin.value,
                self.metadata.elev_map_ymax.value)
    
    def set_robot_pose(self, x: float, y: float, heading: float):
        """Update robot pose (stored in shared memory)"""
        with self.metadata.robot_x.get_lock():
            self.metadata.robot_x.value = x
            self.metadata.robot_y.value = y
            self.metadata.robot_heading.value = heading
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Get robot pose (from shared memory)"""
        return (self.metadata.robot_x.value,
                self.metadata.robot_y.value,
                self.metadata.robot_heading.value)
    
    def get_map_dimensions(self) -> Dict:
        """Get map dimensions"""
        return {
            'elev_map_rows': self.metadata.elev_map_rows.value,
            'elev_map_cols': self.metadata.elev_map_cols.value,
            'trav_map_rows': self.metadata.trav_map_rows.value,
            'trav_map_cols': self.metadata.trav_map_cols.value,
            'trav_map_layers': self.metadata.trav_map_layers.value
        }
    
    # ========================================================================
    # Cleanup
    # ========================================================================
    
    def _signal_handler(self, signum, frame):
        """Handle signals (SIGINT, SIGTERM) - automatic cleanup on crash"""
        self.cleanup()
        sys.exit(0)
    
    def _cleanup_on_exit(self):
        """Cleanup handler for atexit - automatic cleanup on normal exit"""
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up shared memory
        
        Writer: Calls unlink() - removes shared memory from system
        Reader: Calls close() - closes connection (writer handles unlink)
        
        Automatically called on:
        - Normal exit (atexit)
        - SIGINT/SIGTERM signals (signal handlers)
        - Explicit call
        
        Note: SIGKILL cannot be caught, but stale memory is auto-cleaned on next startup
        """
        for name, shm in self.shm_blocks.items():
            if shm:
                try:
                    shm.close()
                    if self.mode == 'writer':
                        shm.unlink()  # Only writer unlinks (removes from system)
                except Exception:
                    pass  # Ignore errors during cleanup
        if self.mode == 'writer':
            print("Shared memory backend cleaned up (writer)")


class LocalMemoryBackend:
    """
    Local memory backend (for testing or single-process mode)
    
    This provides the same interface as SharedMemoryBackend but uses
    regular numpy arrays. Useful for:
    - Testing without shared memory complexity
    - Single-process mode
    - Development/debugging
    """
    
    def __init__(self):
        self.arrays = {}
        self.metadata = {
            'version': 0,
            'is_trav_map_built': False,
            'is_elev_map_built': False,
            'map_resolution': 0.0,
            'robot_x': 0.0,
            'robot_y': 0.0,
            'robot_heading': 0.0
        }
        self._initialized = False
    
    def initialize_arrays(self, elev_map_rows, elev_map_cols, trav_map_rows,
                         trav_map_cols, trav_map_layers, map_resolution,
                         elev_map_xmin, elev_map_xmax, elev_map_ymin, elev_map_ymax):
        """Initialize local arrays"""
        self.arrays['TraversabilityMap'] = np.full(
            (trav_map_rows, trav_map_cols, trav_map_layers, 4), np.nan, dtype=np.float32
        )
        self.arrays['ElevationMap'] = np.full(
            (elev_map_rows, elev_map_cols), np.nan, dtype=np.float32
        )
        self.arrays['ElevMap_MetaInfo'] = np.zeros(
            (elev_map_rows, elev_map_cols), dtype=np.int32
        )
        self.arrays['TravMap_MetaInfo'] = np.zeros(
            (trav_map_rows, trav_map_cols, trav_map_layers), dtype=np.int32
        )
        
        self.metadata['map_resolution'] = map_resolution
        self.metadata['elev_map_xmin'] = elev_map_xmin
        self.metadata['elev_map_xmax'] = elev_map_xmax
        self.metadata['elev_map_ymin'] = elev_map_ymin
        self.metadata['elev_map_ymax'] = elev_map_ymax
        
        self._initialized = True
    
    def get_traversability_map(self) -> np.ndarray:
        return self.arrays['TraversabilityMap']
    
    def get_elevation_map(self) -> np.ndarray:
        return self.arrays['ElevationMap']
    
    def get_elev_meta_info(self) -> np.ndarray:
        return self.arrays['ElevMap_MetaInfo']
    
    def get_trav_meta_info(self) -> np.ndarray:
        return self.arrays['TravMap_MetaInfo']
    
    def acquire_write_lock(self):
        pass  # No-op for local memory
    
    def release_write_lock(self):
        pass
    
    def acquire_read_lock(self):
        pass
    
    def release_read_lock(self):
        pass
    
    def increment_version(self):
        self.metadata['version'] += 1
    
    def get_version(self) -> int:
        return self.metadata['version']
    
    def set_trav_map_built(self, value: bool):
        self.metadata['is_trav_map_built'] = value
    
    def set_elev_map_built(self, value: bool):
        self.metadata['is_elev_map_built'] = value
    
    def get_trav_map_built(self) -> bool:
        return self.metadata['is_trav_map_built']
    
    def get_elev_map_built(self) -> bool:
        return self.metadata['is_elev_map_built']
    
    def get_map_resolution(self) -> float:
        return self.metadata['map_resolution']
    
    def get_map_bounds(self) -> Tuple[float, float, float, float]:
        return (self.metadata['elev_map_xmin'],
                self.metadata['elev_map_xmax'],
                self.metadata['elev_map_ymin'],
                self.metadata['elev_map_ymax'])
    
    def set_robot_pose(self, x: float, y: float, heading: float):
        self.metadata['robot_x'] = x
        self.metadata['robot_y'] = y
        self.metadata['robot_heading'] = heading
    
    def get_robot_pose(self) -> Tuple[float, float, float]:
        return (self.metadata['robot_x'],
                self.metadata['robot_y'],
                self.metadata['robot_heading'])
    
    def cleanup(self):
        pass  # No cleanup needed for local memory

