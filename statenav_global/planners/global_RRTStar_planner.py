


import time
import os
import sys
import math
import numpy as np
from collections import deque
from typing import List, Tuple

from . import BasePlanner
from statenav_global.utility import plotting
from statenav_global.utility import utils


def wrap_to_pi(angle):
    """Wraps an angle to the range [-π, π] using atan2."""
    return np.arctan2(np.sin(angle), np.cos(angle))

# Large integer to represent invalid node index (instead of float('inf'))
INVALID_NODE_IDX = np.iinfo(np.int32).max

class Node(utils.BasicNode):
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]

        self.__cost_trav = 0
        self.__parent = None
        self.__index = None # index of a node = index in self.vertex. index = i -> node = self.vertex[i]

    def set_parent_and_cost(self, parent, cost_trav, index = None):
        self.__parent = parent
        self.__cost_trav = cost_trav
        if index is not None:
            self.__index = index

    def set_index(self, index):
        self.__index = index

    def get_index(self):
        return self.__index

    def reset_index(self):
        self.__index = None

    def get_parent(self):
        return self.__parent
    
    def get_cost_trav(self):
        return self.__cost_trav


class GlobalRRTStar(BasePlanner):
        
    # Constructor Method
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,

        x_start: tuple, 
        x_goal: tuple, 
        heading_start: float,

        goal_radius: float,
        branch_length_max: float, 
        decrease_search_radius: bool,
        search_radius: float, 

        iter_max: int, 
        convergence_threshold: float,
        switch_to_informed_from_thisiter: int,
        sampling_dist: float,
        num_samplingpoints: int,

        default_obstacle_clearance: float,

        ) -> None:
        
        #Initialize BasePlanner
        super().__init__(task_extent, rng)
        self.x_range = (task_extent[0], task_extent[1])
        self.y_range = (task_extent[2], task_extent[3])

        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.heading_start = heading_start
        self.vertex = [self.s_start]
        self.s_start.set_index(0)
        self.path = []

        self.branch_length_max = branch_length_max
        self.goal_radius = goal_radius
        self.search_radius = search_radius

        self.iter_max = iter_max
        self.convergence_threshold = convergence_threshold

        self.decrease_search_radius = decrease_search_radius




        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()
        self.utils.delta = default_obstacle_clearance

        self.obs_circle = []
        self.obs_rectangle = []
        self.obs_boundary = []
    
    



        self.switch_to_informed_from_thisiter =  switch_to_informed_from_thisiter
        self.sampling_dist = sampling_dist
        self.num_samplingpoints = num_samplingpoints


        self.global_map = None
        self.is_goal_reached = False



    def plan(self, heading_c=0):
        
        check_window = self.iter_max // 10
        # path costs is an FIFO queue that has the fized size of check_window
        path_costs = deque(maxlen=check_window)

        self.is_goal_reached = False
        if math.hypot(self.s_start.x - self.s_goal.x, self.s_start.y - self.s_goal.y) < self.goal_radius:
            self.is_goal_reached = True
            print(" Already Reached. No need to plan.")
            print(" Start Node is: ", self.s_start.x, self.s_start.y)

            self.search_goal_parent()
            self.current_iter = 1
            return


        reiterate_counter = 0
        while not self.is_goal_reached and reiterate_counter < 2:
            reiterate_counter += 1

            for k in range(self.iter_max):
                
                self.current_iter = k
                
                if self.current_iter % (self.iter_max // 5) == 0:
                    print(f"\r current iteration: {self.current_iter}. {self.current_iter/self.iter_max*100:.2f}%", end='', flush=True)

                # Generate a random node
                node_rand = self.generate_random_node()
                node_near = self.nearest_neighbor(self.vertex, node_rand) # start from start_node
                node_new = self.new_state(node_near, node_rand)

            
                # Generate a new node and connect it to the tree 
                if node_new:
                    neighbors_indexes = self.find_near_neighbors(node_new) # modify this for rewiring

                    self.vertex.append(node_new)
                    node_new.set_index(len(self.vertex) - 1)

                    if neighbors_indexes:

                        # print("choose parent")
                        self.choose_parent(node_new, neighbors_indexes)
                        # print("rewire")
                        self.rewire(node_new, neighbors_indexes)



                # Check if Converged
                # print("extract path")
                if self.current_iter == self.iter_max - 1:
                    self.is_goal_reached = self.search_goal_parent()
                else:
                    self.is_goal_reached = self.search_goal_parent() #returns False if not reached yet
                if self.is_goal_reached:

                    new_cost_to_goal = self.cost(self.path_vertex[-1]) #TODO: why not -1?

                    # print("new cost to goal is: ", new_cost_to_goal)
                    if new_cost_to_goal == 0:
                        for i in range(1, len(self.path_vertex)):
                            print(' global path vertex:', self.path_vertex[i].x, self.path_vertex[i].y, self.path_vertex[i].get_cost_trav())



                    # add new_cost_to_goal to the queue
                    path_costs.append(new_cost_to_goal)
                    path_costs_np = np.array(path_costs)

                    # if the queue is full, check the convergence
                    if len(path_costs_np) >= check_window:
                        max_cost = np.max(path_costs_np)
                        min_cost = np.min(path_costs_np)
                        
                        # if the difference between the max and min is less than the threshold, break the loop
                        if self.current_iter > self.switch_to_informed_from_thisiter\
                            and (max_cost - min_cost)/max_cost > 0 and (max_cost - min_cost)/max_cost < self.convergence_threshold:
                            print(f"\r Converged. The cost improved by {100*(max_cost - min_cost)/max_cost:.2f}% to {min_cost:.2f}", end='', flush=True)
                            break

                
            
        if not self.is_goal_reached:
            print(" Did not reach the goal")

        




############### RRT* Basic Functions ############################


# adding new nodes

    def generate_random_node(self):
        delta = 0.05 # should be large enough

        if self.switch_to_informed_from_thisiter > 0:

            if self.is_goal_reached and self.current_iter > self.switch_to_informed_from_thisiter:


                # Choose random node in the path to the goal. Exclude the start node
                rand_node_in_path = self.path_vertex[np.random.randint(1, len(self.path_vertex) - 1)]
                rand_node_parent_in_path = rand_node_in_path.get_parent()

                # Construct a square around the random node
                padding = 0.5 * math.hypot(self.s_goal.x - self.s_start.x, self.s_goal.y - self.s_start.y) # might need revision
                padding = padding * ( 0.1 + 1/(1 + self.current_iter - self.switch_to_informed_from_thisiter) )# decrease the padding as the planner converges
                x_min = min(rand_node_in_path.x, rand_node_parent_in_path.x)
                x_max = max(rand_node_in_path.x, rand_node_parent_in_path.x)
                y_min = min(rand_node_in_path.y, rand_node_parent_in_path.y)
                y_max = max(rand_node_in_path.y, rand_node_parent_in_path.y)

                random_node = Node((np.random.uniform(x_min - padding, x_max + padding),
                                    np.random.uniform(y_min - padding, y_max + padding)))
                if random_node.x < self.x_range[0] + delta:
                    random_node.x = self.x_range[0] + delta
                elif random_node.x > self.x_range[1] - delta:
                    random_node.x = self.x_range[1] - delta
                if random_node.y < self.y_range[0] + delta:
                    random_node.y = self.y_range[0] + delta
                elif random_node.y > self.y_range[1] - delta:
                    random_node.y = self.y_range[1] - delta

            else:
                generate = False

                while generate == False:

                    random_node = Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                                np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
                    
                    if self.global_map.is_TraversabilityMap_built:

                        [row, col] = self.global_map.xy2grid(random_node.x, random_node.y)
                        # print("rand_node:", random_node.x, random_node.y)
                        # print("grid index:", row, col)
                        # print("map shape:", self.global_map.TraversabilityMap.shape)
                        trav = np.mean(self.global_map.TraversabilityMap[row, col, :, self.global_map.viz_channel]) #ranged from 0 to 0.5
                        if np.random.uniform(self.global_map.viz_vmin, self.global_map.viz_vmax) < trav:
                            generate = True

                    else:
                        generate = True



        else:
            random_node = Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                        np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))
            
        return random_node



    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_goal):
        
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.branch_length_max, dist) # maximum distance to a new node is branch_length_max\
        
        node_new = Node((node_start.x + dist * math.cos(theta),
                        node_start.y + dist * math.sin(theta)))
        
        if not self.utils.is_collision(node_start, node_new):

            cost_trav = self.cost_travel(node_start=node_start, node_end=node_new)
            node_new.set_parent_and_cost(node_start, cost_trav) # not calculating NN cost yet

            return node_new
        
        else:
            return False


    def find_near_neighbors(self, node_new):
        n = len(self.vertex) + 1
        if self.decrease_search_radius:
            r = self.search_radius * math.sqrt((math.log(n) / n))
        else:
            r = self.search_radius

        dist_table = [math.hypot(node.x - node_new.x, node.y - node_new.y) for node in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]
        
        # 1 to n-1
        # dist_table_index = [ind for ind in range(n-1)]

        return dist_table_index


# rewiring and choosing parent

    # Finding the best parent for newly generated node
    def choose_parent(self, node_new, neighbors_indexes):
        

        cost_list = [self.get_new_cost(node_start=self.vertex[i], node_end=node_new) for i in neighbors_indexes] # cost of nodes of neighbors_indexes

        node_index_mincost = neighbors_indexes[int(np.argmin(cost_list))] # index of cost_min node in self.vertex
        node_mincost_neighbor = self.vertex[node_index_mincost]

        cost_trav = cost_list[int(np.argmin(cost_list))] - self.cost(node_mincost_neighbor) # cost of the cost_min node
        node_new.set_parent_and_cost( parent = node_mincost_neighbor, cost_trav = cost_trav)



    # For neighbors of new_node, check if new_node can be a better parent
    def rewire(self, node_new, neighbors_indexes):
        

        for i in neighbors_indexes:
            node_neighbor = self.vertex[i]
            new_cost = self.get_new_cost(node_start=node_new, node_end=node_neighbor)


            if self.cost(node_neighbor) > new_cost:
                cost_trav = new_cost - self.cost(node_new)
                node_neighbor.set_parent_and_cost(parent = node_new, cost_trav = cost_trav)


                # self.rewire(node_neighbor, self.find_near_neighbors(node_neighbor))





############### Cost-Related ############################





    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    

    def cost_travel(self, node_start, node_end):
        """
        cost to travel from node_start to node_end
        """

        cost_travel = 0
        dist = math.hypot(node_start.x - node_end.x, node_start.y - node_end.y)
        # dist = math.sqrt((node_start.x - node_end.x)**2 + (node_start.y - node_end.y)**2)
        # cost_travel = dist/0.5

        if node_start.get_parent() is None:
            c1 = self.heading_start
        else:
            c1 = np.arctan2(node_start.y - node_start.get_parent().y, node_start.x - node_start.get_parent().x)

        if self.current_iter < self.switch_to_informed_from_thisiter:
            edge_cost = self.global_map.get_edge_cost(node_start.x, node_start.y, c1, node_end.x, node_end.y, mode = 'SamplingInEdge', sampling_distance = self.sampling_dist, num_interpoints = self.num_samplingpoints)
        else:
            edge_cost = self.global_map.get_edge_cost(node_start.x, node_start.y, c1, node_end.x, node_end.y, mode = 'SamplingInEdge', sampling_distance = 0.04, num_interpoints = self.num_samplingpoints)
    


        cost_travel = edge_cost

        




        return cost_travel



    def cost(self, node_p):
        """
        entire cost to reach this node from the root node
        """

        node = node_p
        cost = 0.0

        while node.get_parent() is not None:
            cost += node.get_cost_trav()
            node = node.get_parent()

            

        if node != self.s_start:
            raise ValueError("Node_p is not connected to the root node")

        return cost
    


    def get_new_cost(self, node_start, node_end): 
        """
        get new cost for node_end if it has node_start as its parent
        """
        return self.cost(node_start) + self.cost_travel(node_start, node_end)
    








############################ Utility Functions and Post-Processing ########################################


    def search_goal_parent(self):
        """
        Search the goal parent node based on path cost.
        Extract the path.
        Return True if the goal is reached, False otherwise.

        if self.obstacles_list is not None:
            Calculate the object cost of the path passing through the obstacles.
            Find the best path based on it.
        """

        # Trivial case: the goal is already reached.
        if math.hypot(self.s_start.x - self.s_goal.x, self.s_start.y - self.s_goal.y) < self.goal_radius:
            # The goal is already reached.
            self.s_goal.set_parent_and_cost(parent = self.s_start, cost_trav = self.cost_travel(node_start=self.s_start, node_end=self.s_goal))
            self.path.append([self.s_goal.x, self.s_goal.y])
            self.path_vertex.append(self.s_goal)
            self.path.append([self.s_start.x, self.s_start.y])
            self.path_vertex.append(self.s_start)
            return True




        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        neargoal_node_indexes = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_radius]
        
        if len(neargoal_node_indexes) > 0:
            cost_list = [self.cost(self.vertex[i]) for i in neargoal_node_indexes
                            if not self.utils.is_collision(self.vertex[i], self.s_goal)]




            # Extract the best path
            if len(cost_list) > 0:
                MinCostNode_IdxTotal = neargoal_node_indexes[int(np.argmin(cost_list))]
                MinCostNode = self.vertex[ MinCostNode_IdxTotal ]
                self.s_goal.set_parent_and_cost(parent = MinCostNode, cost_trav = self.cost_travel(node_start=MinCostNode, node_end=self.s_goal))

                self.path = []
                self.path_vertex = []
                node = self.s_goal
                while node.get_parent() is not None:

                    # path extraction
                    self.path.append([node.x, node.y])
                    self.path_vertex.append(node)
                    node = node.get_parent()

                self.path.append([node.x, node.y])
                self.path_vertex.append(node)

                self.path.reverse()
                self.path_vertex.reverse()
                

                
                return True

                
        else:
            return False


    

        

    def reset_tree(self, start:tuple=None, goal:tuple=None, heading_start:float=None):
        
        if start is not None:
            self.s_start = Node(start)
            
        if goal is not None:
            self.s_goal = Node(goal)

        if heading_start is not None:
            self.heading_start = heading_start
        
        self.path = []
        self.vertex = [self.s_start]
        self.s_start.set_index(0)
        self.path_vertex = []
        self.plotting.xI = (self.s_start.x, self.s_start.y)
        self.plotting.xG = (self.s_goal.x, self.s_goal.y)

    def replan(self, initial_start=None, step_T=0.4, RRT_getwaypoint_steps=10, plot_map=False):


        if self.global_map.is_TraversabilityMap_built:
            
            # ==================================== Global Planning ====================================

            start_time = time.time()
            print("\nReplanning Global Path!")
            print(" Previous global start was: ", np.round(self.s_start.x,3), np.round(self.s_start.y,3))
            print(" Current Robot Position: ", np.round(self.global_map.robot_x,3), np.round(self.global_map.robot_y,3), np.round(np.rad2deg(self.global_map.robot_heading)))
            
            [cmd_v_limit, cmd_w_limit] = self.global_map.get_cmd_limits()
            waypoint = self.global_map.get_waypoint(self.global_map.robot_x, self.global_map.robot_y, self.global_map.robot_heading, step_T, RRT_getwaypoint_steps, cmd_v_limit, cmd_w_limit, self.path)
            print(" Next Waypoint: ", waypoint)

            if (waypoint is not False)\
                and not (waypoint[0] == self.s_goal.x or waypoint[1] == self.s_goal.y)\
                    and not (self.s_start.x == initial_start[0] and self.s_start.y == initial_start[1]): # robot position was not updated
                self.reset_tree(start=(waypoint[0], waypoint[1]), heading_start=np.arctan2(waypoint[1] - self.global_map.robot_y, waypoint[0] - self.global_map.robot_x))
            else:
                self.reset_tree(start=(self.global_map.robot_x, self.global_map.robot_y),  heading_start=self.global_map.robot_heading)

            print(" The start is: ", np.round(self.s_start.x,3), np.round(self.s_start.y,3), np.round(np.rad2deg(self.heading_start),3))
            print(" The goal is: ", np.round(self.s_goal.x, 3), np.round(self.s_goal.y, 3))

            self.plan(heading_c=self.global_map.robot_heading)
            print('\n RRT* Computation Time took:', np.round(time.time() - start_time,3), 's') 
            print(f' Total Navigation Cost (Estimated Traversal Time): {np.round(self.cost(self.s_goal),3)} s')
            print("Global Path Replanned!\n")

            if plot_map:
                self.plot_map()








    def plot_map(self):
        
        if self.path:
            self.plotting.animation_interactive(self.vertex, self.path, "RRT*, Total Iterations = " + str(self.current_iter)+ ", ETA = " + str(np.round(self.cost(self.s_goal), 1)), )
        else:
            self.plotting.plot_grid("Plain Environment")






            
        


#################### RRT* additional functions ############################
    
    def update_static_obstacles(self, obs_cir=[], obs_bound=[], obs_rec=[], obstacles=[]):
        """
        Update obstacle representation by determining minimal rectangles to cover circular obstacles.
        Parameters:
            obs_cir (list): List of circular obstacles, each given as (x, y, radius).
            obs_bound (list): Placeholder for boundary obstacles (not used here).
            obs_rec (list): Output list to store rectangular obstacles as (x, y, length, width).
            obstacles (list): Additional obstacles (not used here).
        """


            


    
        self.obs_circle = obs_cir
        self.obs_boundary = obs_bound
        self.obs_rectangle = obs_rec
        # self.obstacle = obstacles
        
        self.plotting.obs_circle = self.obs_circle
        self.plotting.obs_boundary = self.obs_boundary
        self.plotting.obs_rectangle = self.obs_rectangle
        # self.plotting.obstacle = obstacles
        
        self.utils.obs_circle = self.obs_circle
        self.utils.obs_boundary = self.obs_boundary
        self.utils.obs_rectangle = self.obs_rectangle
        # self.utils.obstacle = obstacles
            
            