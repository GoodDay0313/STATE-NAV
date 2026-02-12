"""
Plotting tools for Sampling-based algorithms
@author: huiming zhou
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")


class Plotting:
    def __init__(self, x_start, x_goal):
        self.xI, self.xG = x_start, x_goal
        self.obs_bound = []
        self.obs_circle = []
        self.obs_rectangle = []
        self.obstacle = []

    def animation(self, nodelist, path, name, animation=False):
        self.plot_grid(name)
        self.plot_visited(nodelist, animation)
        self.plot_path(path)
    def animation_connect(self, V1, V2, path, name):
        self.plot_grid(name)
        self.plot_visited_connect(V1, V2)
        self.plot_path(path)


    def animation_interactive(self, nodelist, path, name, animation=False, nodes_before_obstacles=[], nodes_after_obstacles=[]):
        self.plot_grid_interactive(name)
        self.plot_visited_interactive(nodelist, animation)
        self.plot_path_interactive(path)

        if len(nodes_before_obstacles) > 0 and len(nodes_after_obstacles) > 0:
            self.plot_obstacle_nodes(nodes_before_obstacles, nodes_after_obstacles)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_grid_interactive(self, name):
        
        plt.close(4)
        plt.figure(4)
        # plt.ion()
        self.fig, self.ax = plt.subplots(num=4)
        manager = self.fig.canvas.manager
        manager.window.setGeometry(1440, 0, 480, 400)  # Adjust the position and size as needed

        for (ox, oy, w, h) in self.obs_bound:
            self.ax.add_patch(
                patches.Rectangle(
                    (oy, ox), h, w,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (oy, ox), h, w,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True,
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (oy, ox), r,
                    edgecolor='black',
                    facecolor='red', alpha=0.3
                )
            )
            
        for path in self.obstacle:
            patch = patches.PathPatch(path, facecolor='gray', lw=1)
            self.ax.add_patch(patch)

        self.ax.plot(self.xI[1], self.xI[0], "bs", linewidth=0.5, markersize=4) # blue square in the beginning
        self.ax.plot(self.xG[1], self.xG[0], "gs", linewidth=0.5, markersize=4) # green square in the end

        self.ax.set_title(name, fontsize=15)
        
        # Invert x and y axes
        self.ax.invert_xaxis()
        # self.ax.invert_yaxis()
        
        # Add grid
        self.ax.grid(True)
        
        self.ax.set_xlabel('Y Axis')
        self.ax.set_ylabel('X Axis')
        
    def plot_visited_interactive(self, nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.get_parent():
                    self.ax.plot([node.get_parent().x, node.x], [node.get_parent().y, node.y], "-g", linewidth=0.5) # green line thcikness add 1 
                    self.ax.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:

                self.ax.plot(node.y, node.x, "Db", markersize=1.7) # blue square size
                if node.get_parent():
                    self.ax.plot([node.get_parent().y, node.y], [node.get_parent().x, node.x], "-g", linewidth=0.5) # green line thcikness add 1


    def plot_path_interactive(self, path):
        if len(path) != 0:
            self.ax.plot([x[1] for x in path], [x[0] for x in path], "Dr", markersize=2) # red square size on the path
            self.ax.plot([x[1] for x in path], [x[0] for x in path], '-r', linewidth=1) # red line on the path
            plt.pause(0.001)

    def plot_obstacle_nodes(self, nodes_before_obstacles, nodes_after_obstacles):
        """
        Plot node_before_obstacle and node_after_obstacle markers on the current plot.
        
        Parameters
        ----------
        nodes_before_obstacles : list
            List of nodes before obstacles
        nodes_after_obstacles : list
            List of nodes after obstacles
        """
        if len(nodes_before_obstacles) > 0 and len(nodes_after_obstacles) > 0:
            for idx in range(len(nodes_before_obstacles)):
                node_before = nodes_before_obstacles[idx]
                node_after = nodes_after_obstacles[idx]
                
                # Plot node_before_obstacle in magenta
                self.ax.plot(node_before.y, node_before.x, "c*", markersize=10, label="Before Obstacle" if idx == 0 else "")
                # Plot node_after_obstacle in cyan
                self.ax.plot(node_after.y, node_after.x, "c*", markersize=10, label="After Obstacle" if idx == 0 else "")
            

        
    def plot_grid(self, name):
        self.fig, ax = plt.subplots(figsize=(40,20))

        for (ox, oy, w, h) in self.obs_bound:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True,
                )
            )

        for (ox, oy, r) in self.obs_circle:
            ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='red', alpha=0.3
                )
            )
            
        for path in self.obstacle:
            patch = patches.PathPatch(path, facecolor='gray', lw=1)
            ax.add_patch(patch)

        ax.plot(self.xI[0], self.xI[1], "bs", linewidth=3)
        ax.plot(self.xG[0], self.xG[1], "gs", linewidth=3)

        ax.set_title(name)

    @staticmethod
    def plot_visited(nodelist, animation):
        if animation:
            count = 0
            for node in nodelist:
                count += 1
                if node.get_parent():
                    plt.plot([node.get_parent().x, node.x], [node.get_parent().y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in nodelist:
                if node.get_parent():
                    plt.plot([node.get_parent().x, node.x], [node.get_parent().y, node.y], "-g")

    @staticmethod
    def plot_visited_connect(V1, V2):
        len1, len2 = len(V1), len(V2)

        for k in range(max(len1, len2)):
            if k < len1:
                if V1[k].get_parent():
                    plt.plot([V1[k].x, V1[k].get_parent().x], [V1[k].y, V1[k].get_parent().y], "-g")
            if k < len2:
                if V2[k].get_parent():
                    plt.plot([V2[k].x, V2[k].get_parent().x], [V2[k].y, V2[k].get_parent().y], "-g")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 2 == 0:
                plt.pause(0.001)

        plt.pause(0.01)

    @staticmethod
    def plot_path(path):
        if len(path) != 0:
            plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
            plt.pause(0.01)
        plt.show(block=False)





