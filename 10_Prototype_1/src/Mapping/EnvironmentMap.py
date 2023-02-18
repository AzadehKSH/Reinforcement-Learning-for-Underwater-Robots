from Mapping.Octree import Octree
from Mapping.BoundingBox import BoundingBox
from Mapping.Cell import Cell, CellLabel
from Mapping.PathPlanning import RRT

class EnvironmentMap:
    """Data structure that stores all sensor data of the agent and keeps track of the agent's next target point.
    """
    def __init__(self, collision_resolution, covered_resolution, unexplored_resolution):
        """Data structure that stores all sensor data of the agent and keeps track of the agent's next target point.

        :param collision_resolution: Tuple of the x-, y- and z-resolution of the octree.
        :type collision_resolution: Tuple(Float, Float, Float)
        :param covered_resolution: Tuple of the x-, y- and z-resolution of the octree.
        :type covered_resolution: Tuple(Float, Float, Float)
        :param unexplored_resolution: Tuple of the x-, y- and z-resolution of the octree.
        :type unexplored_resolution: Tuple(Float, Float, Float)
        """
        self.collision_points: Octree = Octree(collision_resolution, BoundingBox((0,0,0), (0,0,0)))
        self.covered_points: Octree = Octree(covered_resolution, BoundingBox((0,0,0), (0,0,0)))
        self.unexplored_points: Octree = Octree(unexplored_resolution, BoundingBox((0,0,0), (0,0,0)))
        self.boundary: BoundingBox = BoundingBox((float("inf"), float("inf"), float("inf")), (-float("inf"), -float("inf"), -float("inf")))
        self._path_planning = None
        
    def _generate_new_quadtree(self, boundary_old, boundary_new, resolution):
        boundary = BoundingBox(
            (
                min(boundary_old[0][0], boundary_new[0][0]),
                min(boundary_old[0][1], boundary_new[0][1]),
                min(boundary_old[0][2], boundary_new[0][2])
            ),
            (
                max(boundary_old[1][0], boundary_new[1][0]),
                max(boundary_old[1][1], boundary_new[1][1]),
                max(boundary_old[1][2], boundary_new[1][2])
            )
        )

        return Octree(resolution, boundary)

    def update_collision_points(self, pointcloud, boundary: BoundingBox):
        """Add new collision points to the collision_points octree.

        :param pointcloud: List of collision points.
        :type pointcloud: List(Cell)
        :param boundary: Bounding box that covers all points of the pointcloud.
        :type boundary: BoundingBox
        """
        collision_points_elems = self.collision_points.get_all_cells()
        points = collision_points_elems + [Cell(point[0], point[1], point[2], CellLabel.COLLISION) for point in pointcloud]
        
        collision_points = self._generate_new_quadtree(
            self.collision_points.get_boundary().get_bounding_box(), 
            boundary.get_bounding_box(), 
            self.collision_points.get_resolution()
        )

        for point in points:
            collision_points.insert(point)
        
        min_x = collision_points.get_boundary().get_bounding_box()[0][0] if self.boundary.get_bounding_box()[0][0] > collision_points.get_boundary().get_bounding_box()[0][0] else self.boundary.get_bounding_box()[0][0]
        min_y = collision_points.get_boundary().get_bounding_box()[0][1] if self.boundary.get_bounding_box()[0][1] > collision_points.get_boundary().get_bounding_box()[0][1] else self.boundary.get_bounding_box()[0][1]
        min_z = collision_points.get_boundary().get_bounding_box()[0][2] if self.boundary.get_bounding_box()[0][2] > collision_points.get_boundary().get_bounding_box()[0][2] else self.boundary.get_bounding_box()[0][2]

        max_x = collision_points.get_boundary().get_bounding_box()[1][0] if self.boundary.get_bounding_box()[1][0] < collision_points.get_boundary().get_bounding_box()[1][0] else self.boundary.get_bounding_box()[1][0]
        max_y = collision_points.get_boundary().get_bounding_box()[1][1] if self.boundary.get_bounding_box()[1][1] < collision_points.get_boundary().get_bounding_box()[1][1] else self.boundary.get_bounding_box()[1][1]
        max_z = collision_points.get_boundary().get_bounding_box()[1][2] if self.boundary.get_bounding_box()[1][2] < collision_points.get_boundary().get_bounding_box()[1][2] else self.boundary.get_bounding_box()[1][2]

        self.boundary = BoundingBox((min_x, min_y, min_z), (max_x, max_y, max_z))
        self.collision_points = collision_points

    def update_covered_points(self, pointcloud, boundary: BoundingBox):
        """Add new covered points to the covered_points octree.

        :param pointcloud: List of covered points.
        :type pointcloud: List(Cell)
        :param boundary: Bounding box that covers all points of the pointcloud.
        :type boundary: BoundingBox
        """
        covered_points_elems = self.covered_points.get_all_cells()
        points = covered_points_elems + [Cell(point[0], point[1], point[2], CellLabel.COVERED) for point in pointcloud]
        
        covered_points = self._generate_new_quadtree(
            self.covered_points.get_boundary().get_bounding_box(), 
            boundary.get_bounding_box(), 
            self.covered_points.get_resolution()
        )

        for point in points:
            covered_points.insert(point)
        
        min_x = covered_points.get_boundary().get_bounding_box()[0][0] if self.boundary.get_bounding_box()[0][0] > covered_points.get_boundary().get_bounding_box()[0][0] else self.boundary.get_bounding_box()[0][0]
        min_y = covered_points.get_boundary().get_bounding_box()[0][1] if self.boundary.get_bounding_box()[0][1] > covered_points.get_boundary().get_bounding_box()[0][1] else self.boundary.get_bounding_box()[0][1]
        min_z = covered_points.get_boundary().get_bounding_box()[0][2] if self.boundary.get_bounding_box()[0][2] > covered_points.get_boundary().get_bounding_box()[0][2] else self.boundary.get_bounding_box()[0][2]
        
        max_x = covered_points.get_boundary().get_bounding_box()[1][0] if self.boundary.get_bounding_box()[1][0] < covered_points.get_boundary().get_bounding_box()[1][0] else self.boundary.get_bounding_box()[1][0]
        max_y = covered_points.get_boundary().get_bounding_box()[1][1] if self.boundary.get_bounding_box()[1][1] < covered_points.get_boundary().get_bounding_box()[1][1] else self.boundary.get_bounding_box()[1][1]
        max_z = covered_points.get_boundary().get_bounding_box()[1][2] if self.boundary.get_bounding_box()[1][2] < covered_points.get_boundary().get_bounding_box()[1][2] else self.boundary.get_bounding_box()[1][2]
        
        self.boundary = BoundingBox((min_x, min_y, min_z), (max_x, max_y, max_z))

        self.covered_points = covered_points

    def update_unexplored_points(self, pointcloud, boundary: BoundingBox, rob_pos):
        """Add new unexplored points to the unexplored_points octree.

        :param pointcloud: List of unexplored points.
        :type pointcloud: List(Cell)
        :param boundary: Bounding box that covers all points of the pointcloud.
        :type boundary: BoundingBox
        """
        # IMPROVEMENTS: reimplement this method
        if len(pointcloud) == 0:
            return
        rob_bbox = BoundingBox(
            (rob_pos[0] - self.covered_points.get_resolution()[0], rob_pos[1] - self.covered_points.get_resolution()[1], rob_pos[2] - self.covered_points.get_resolution()[2]),
            (rob_pos[0] + self.covered_points.get_resolution()[0], rob_pos[1] + self.covered_points.get_resolution()[1], rob_pos[2] + self.covered_points.get_resolution()[2])
        )
        unexplored_points_elems = self.unexplored_points.get_all_cells()
        all_points = [Cell(point[0], point[1], point[2], CellLabel.UNEXPLORED) for point in pointcloud] + unexplored_points_elems
        points = [cell for cell in all_points if 
                  not rob_bbox.contains_cell(cell) and
                  len(self.collision_points.find_cells(
                      BoundingBox(
                          (cell.x - self.collision_points.get_resolution()[0], cell.y - self.collision_points.get_resolution()[1], cell.z -self.collision_points.get_resolution()[2]),
                          (cell.x + self.collision_points.get_resolution()[0], cell.y + self.collision_points.get_resolution()[1], cell.z + self.collision_points.get_resolution()[2])
                      )
        )) == 0 and # no collision points in the vicinity of the unexp. point otherwise a collision with the point is to be expected
        len(self.covered_points.find_cells(
            BoundingBox(
                (cell.x - self.covered_points.get_resolution()[0], cell.y - self.covered_points.get_resolution()[1], cell.z -self.covered_points.get_resolution()[2]),
                (cell.x + self.covered_points.get_resolution()[0], cell.y + self.covered_points.get_resolution()[1], cell.z + self.covered_points.get_resolution()[2])
            )
        )) < 5 # less than 5 explored points in the vicinity of the unexp. point
        ]
        # Currently points are unexplored iff less than 10 explored points are in the near. However, a point shall be explored if at least one point in the near surroundings. This currently breaks the implementation due to the nature how unexp. points are calculated. Need to reimplement unexp. points, e.g. get rid of unexp. points at all and calculate it using explored points adhoc.
        del rob_bbox
        
        unexplored_points = self._generate_new_quadtree(
            self.unexplored_points.get_boundary().get_bounding_box(), 
            boundary.get_bounding_box(), 
            self.unexplored_points.get_resolution()
        )

        for point in points:
            unexplored_points.insert(point)
        unexplored_points.build_kd_tree()

        min_x = unexplored_points.get_boundary().get_bounding_box()[0][0] if self.boundary.get_bounding_box()[0][0] > unexplored_points.get_boundary().get_bounding_box()[0][0] else self.boundary.get_bounding_box()[0][0]
        min_y = unexplored_points.get_boundary().get_bounding_box()[0][1] if self.boundary.get_bounding_box()[0][1] > unexplored_points.get_boundary().get_bounding_box()[0][1] else self.boundary.get_bounding_box()[0][1]
        min_z = unexplored_points.get_boundary().get_bounding_box()[0][2] if self.boundary.get_bounding_box()[0][2] > unexplored_points.get_boundary().get_bounding_box()[0][2] else self.boundary.get_bounding_box()[0][2]

        max_x = unexplored_points.get_boundary().get_bounding_box()[1][0] if self.boundary.get_bounding_box()[1][0] < unexplored_points.get_boundary().get_bounding_box()[1][0] else self.boundary.get_bounding_box()[1][0]
        max_y = unexplored_points.get_boundary().get_bounding_box()[1][1] if self.boundary.get_bounding_box()[1][1] < unexplored_points.get_boundary().get_bounding_box()[1][1] else self.boundary.get_bounding_box()[1][1]
        max_z = unexplored_points.get_boundary().get_bounding_box()[1][2] if self.boundary.get_bounding_box()[1][2] < unexplored_points.get_boundary().get_bounding_box()[1][2] else self.boundary.get_bounding_box()[1][2]

        self.boundary = BoundingBox((min_x, min_y, min_z), (max_x, max_y, max_z))
        
        self.unexplored_points = unexplored_points

    def generate_path(self, robot_coordinates_x : float, robot_coordinates_y : float, robot_coordinates_z : float):
        """Generates a path from the robot's position to its nearest neighbour.

        :param robot_coordinates_x: x-coordinate of the robot's position
        :type robot_coordinates_x: float
        :param robot_coordinates_y: y-coordinate of the robot's position
        :type robot_coordinates_y: float
        :param robot_coordinates_z: z-coordinate of the robot's position
        :type robot_coordinates_z: float
        :return: A path from the robot's position to the nearest neighbour. 
        :rtype: List(Tuple(Float, Float, Float))
        """
        if self.unexplored_points.isEmpty():
            return []
        
        self._path_planning = RRT(boundary=self.boundary)

        start = Cell(robot_coordinates_x, robot_coordinates_y, robot_coordinates_z, 0)
        goal = self.unexplored_points.find_nearest_neighbour(start)
        
        path = self._path_planning.generate_path(start, goal, self.collision_points)
            
        return path
        
