from random import Random
from re import X
from turtle import st
from matplotlib.colors import same_color

from matplotlib.pyplot import step
from Mapping.Cell import Cell, CellLabel
from Mapping.QuadTree import Quadtree
from Mapping.BoundingBox import BoundingBox

import numpy as np
import math

class Node(Cell):
    def __init__(self, x, y, label = 0, parent = None):
        self.parent = parent

class RRT:
    def __init__(self, boundary: BoundingBox, sample_rate : float = 0.05, step_size : float = 0.25, seed=1337, max_iteration=100):
        self.nodes = []
        self.obstacles : Quadtree = None
        self._max_iteration = max_iteration
        self._step_size = step_size
        self._boundary : BoundingBox = boundary
        self._sample_rate = sample_rate

        self.start = None
        self.goal = None

        np.random.seed(seed)

    def generate_path(self, start : Cell, goal : Cell, obstacles : Quadtree):
        self.start = Node(start.x, start.y, start.label)
        self.goal = Node(goal.x, goal.y, goal.label)

        self.obstacles = obstacles
        self.nodes.append(self.start)

        direct = self._get_direct_path(self.start, self.goal)

        if len(direct) > 0:
            for node in direct:
                self.nodes.append(node)
            return self.extract_path(direct[-1])

        for _ in range(self._max_iteration):
            direct = []
            random_node = self._generate_random_node()
            nearest_node = self._nearest_neighbour(random_node)
            new_node = self._calc_new_state(nearest_node, random_node)

            if new_node and not RRT._has_collision(new_node, self.obstacles):
                self.nodes.append(new_node)

                direct = self._get_direct_path(new_node, self.goal)
                if len(direct) > 0:
                    for node in direct:
                        self.nodes.append(node)
                    return self.extract_path(direct[-1])
                
                dist, _ = self._get_distance_and_angle(new_node, self.goal)
                print (str(dist) + ":" + str(dist <= self._step_size))
                if dist <= self._step_size and not self._has_collision(new_node, self.obstacles):
                    new_node = self._calc_new_state(new_node, self.goal)
                    print(new_node)
                    return self.extract_path(new_node)

    def _get_direct_path(self, start : Node, goal : Node):
        distance = math.dist((start.x, start.y), (goal.x, goal.y))
        t = 0
        result = []
        while t < 1.:
            t += self._step_size/distance
            x = (1. - t) * start.x + t * goal.x
            y = (1. - t) * start.y + t * goal.y

            node = Node(x,y,0)
            node.parent = start if len(result) == 0 else result[-1]
            
            if self._has_collision(node, self.obstacles):
                return []
            result.append(node)
        return result


    def _generate_random_node(self):
        if np.random.random() > self._sample_rate:
            return Node(
                np.random.uniform(self._boundary.get_bounding_box()[0][0] - self._step_size, self._boundary.get_bounding_box()[1][0] - self._step_size),
                np.random.uniform(self._boundary.get_bounding_box()[0][1] + self._step_size, self._boundary.get_bounding_box()[1][1] + self._step_size), 0
            )
        return self.goal
    
    def _nearest_neighbour(self, node):
        return self.nodes[
            int(np.argmin([math.hypot(nd.x - node.x, nd.y - node.y) for nd in self.nodes]))
        ]
    
    def _get_distance_and_angle(self, start : Node, goal : Node):
        dx = start.x - goal.x
        dy = start.y - goal.y

        return math.hypot(dx, dy), math.atan2(dy, dx)

    def _calc_new_state(self, start : Node, goal : Node):
        dist, theta = self._get_distance_and_angle(start, goal)

        dist = min(self._step_size, dist)
        new_node = Node(
            start.x + dist * math.cos(theta),
            start.y + dist * math.sin(theta),
            0
        )

        new_node.parent = start

        return new_node

    def extract_path(self, node_end : Node):
        path = [(self.goal.x, self.goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))
        
        return path
    
    @staticmethod
    def _has_collision(node : Node, collisions : Quadtree):
        nearest_neighbour = collisions.find_nearest_neighbour(node)
        if nearest_neighbour is None:
            return False
        
        if BoundingBox(
            (nearest_neighbour.x - collisions.get_resolution()[0], nearest_neighbour.y - collisions.get_resolution()[1]),
            (nearest_neighbour.x + collisions.get_resolution()[0], nearest_neighbour.y + collisions.get_resolution()[1])
            ).contains_cell(node):
            return True
        return False



    

