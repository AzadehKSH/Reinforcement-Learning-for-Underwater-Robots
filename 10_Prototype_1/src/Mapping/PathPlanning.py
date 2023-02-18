from random import Random
from re import X
from matplotlib.colors import same_color

from matplotlib.pyplot import step
from Mapping.Cell import Cell, CellLabel
from Mapping.Octree import Octree
from Mapping.BoundingBox import BoundingBox

import numpy as np
import math

class Node(Cell):
    """Representation of a point for the path generation algorithms. Inherits the Cell class.
    """
    def __init__(self, x, y, z, label = 0, parent = None):
        """Representation of a point for the path generation algorithms. Inherits the Cell class.

        :param x: x-coordinate of the point
        :type x: Float
        :param y: y-coordinate of the point
        :type y: Float
        :param z: z-coordinate of the point
        :type z: Float
        :param label: Label of the node, defaults to 0
        :type label: CellLabel, optional
        :param parent: Parent node, defaults to None
        :type parent: Node, optional
        """
        self.parent = parent

class RRT:
    """Implements a modified RRT algorithm. Beside using the normal RRT the algorithm tries to draw a direct line between the current node and the goal node. 
    """
    def __init__(self, boundary: BoundingBox, sample_rate : float = 0.05, step_size : float = 0.25, seed=1337, max_iteration=100):
        """Implements a modified RRT algorithm. Beside using the normal RRT the algorithm tries to draw a direct line between the current node and the goal node. 

        :param boundary: The environment map's bounding box.
        :type boundary: BoundingBox
        :param sample_rate: Likelihood, that the next node is the goal node. Defaults to 0.05
        :type sample_rate: Float, optional
        :param step_size: Step size between to nodes, defaults to 0.25
        :type step_size: Float, optional
        :param seed: A random seed for determinism, defaults to 1337
        :type seed: Int, optional
        :param max_iteration: Maximal number of iterations, defaults to 100
        :type max_iteration: Int, optional
        """
        self.nodes = []
        self.obstacles : Octree = None
        self._max_iteration = max_iteration
        self._step_size = step_size
        self._boundary : BoundingBox = boundary
        self._sample_rate = sample_rate

        self.start = None
        self.goal = None

        np.random.seed(seed)

    def generate_path(self, start : Cell, goal : Cell, obstacles : Octree):
        """Generates a path from the start to the goal cell.

        :param start: Start cell.
        :type start: Cell
        :param goal: Goal cell.
        :type goal: Cell
        :param obstacles: Collision points. These point cannot be overcome.
        :type obstacles: Octree
        :return: List of points from the start to the goal point. 
        :rtype: List(Tuple(Float, Float, Float))
        """
        
        self.start = Node(start.x, start.y, start.z, start.label)
        self.goal = Node(goal.x, goal.y, goal.z, goal.label)

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
                # print(f"new node: {new_node}")

                direct = self._get_direct_path(new_node, self.goal)
                if len(direct) > 0:
                    for node in direct:
                        self.nodes.append(node)
                    return self.extract_path(direct[-1])
                
                dist = self._get_distance(new_node, self.goal)
                # print (str(dist) + ":" + str(dist <= self._step_size))
                if dist <= self._step_size and not self._has_collision(new_node, self.obstacles):
                    new_node = self._calc_new_state(new_node, self.goal)
                    # print(new_node)
                    return self.extract_path(new_node)

    def _get_direct_path(self, start : Node, goal : Node):
        distance = self._get_distance(start, goal)
        t = 0
        result = []
        while t < 1.:
            t += self._step_size/distance
            x = (1. - t) * start.x + t * goal.x
            y = (1. - t) * start.y + t * goal.y
            z = (1. - t) * start.z + t * goal.z

            node = Node(x,y,z,0)
            node.parent = start if len(result) == 0 else result[-1]
            if self._has_collision(node, self.obstacles):
                return []
            result.append(node)
        return result


    def _generate_random_node(self):
        if np.random.random() > self._sample_rate:
            return Node(
                np.random.uniform(self._boundary.get_bounding_box()[0][0] + self._step_size, self._boundary.get_bounding_box()[1][0] - self._step_size),
                np.random.uniform(self._boundary.get_bounding_box()[0][1] + self._step_size, self._boundary.get_bounding_box()[1][1] - self._step_size), 
                np.random.uniform(self._boundary.get_bounding_box()[0][2] + self._step_size, self._boundary.get_bounding_box()[1][2] - self._step_size),
                0
            )
        return self.goal
    
    def _nearest_neighbour(self, node):
        return self.nodes[
            int(np.argmin([math.hypot(nd.x - node.x, nd.y - node.y, nd.z - node.z) for nd in self.nodes]))
        ]
    
    def _get_distance(self, start : Node, goal : Node):
        dx = start.x - goal.x
        dy = start.y - goal.y
        dz = start.z - goal.z

        return math.hypot(dx, dy, dz)

    def _calc_new_state(self, start : Node, goal : Node):
        dist = self._get_distance(start, goal)

        step = min(self._step_size, dist)
        increment = ((goal.x - start.x) / dist * step, (goal.y - start.y) / dist * step, (goal.z - start.z) / dist * step)
        new_node = Node(
            start.x + increment[0],
            start.y + increment[1],
            start.z + increment[2],
            0
        )

        new_node.parent = start

        return new_node

    def extract_path(self, node_end : Node):
        """Extract the path if a path was found.

        :param node_end: The current node.
        :type node_end: Node
        :return: Path from the current to the goal point.
        :rtype: List(Tuple(Float, Float, Float))
        """
        path = [(self.goal.x, self.goal.y, self.goal.z)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y, node_now.z))
        
        return path
    
    @staticmethod
    def _has_collision(node : Node, collisions : Octree):
        bbox = BoundingBox(
            (node.x - collisions.get_resolution()[0], node.y - collisions.get_resolution()[1], node.z - collisions.get_resolution()[2]),
            (node.x + collisions.get_resolution()[0], node.y + collisions.get_resolution()[1], node.z + collisions.get_resolution()[2])
        )
        
        collision = collisions.find_cells(bbox)

        if len(collision) == 0:
            return False
        return True