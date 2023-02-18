from Mapping.QuadTree import Quadtree
from Mapping.BoundingBox import BoundingBox
from Mapping.Cell import Cell, CellLabel
from Mapping.PathPlanning import RRT

class EnvironmentMap:
    def __init__(self, collision_resolution, covered_resolution, unexplored_resolution):
        self.collision_points: Quadtree = Quadtree(collision_resolution, BoundingBox((0,0), (0,0)))
        self.covered_points: Quadtree = Quadtree(covered_resolution, BoundingBox((0,0), (0,0)))
        self.unexplored_points: Quadtree = Quadtree(unexplored_resolution, BoundingBox((0,0), (0,0)))
        self.boundary: BoundingBox = BoundingBox((float("inf"), float("inf")), (-float("inf"), -float("inf")))
        self.scan = []
        self._path_planning = None
    
    def update_depth_scan(self, pointcloud):
        self.scan += pointcloud
        
    def _generate_new_quadtree(self, boundary_old, boundary_new, resolution):
        boundary = BoundingBox(
            (
                min(boundary_old[0][0], boundary_new[0][0]),
                min(boundary_old[0][1], boundary_new[0][1])
            ),
            (
                max(boundary_old[1][0], boundary_new[1][0]),
                max(boundary_old[1][1], boundary_new[1][1])
            )
        )

        return Quadtree(resolution, boundary)

    def update_collision_points(self, pointcloud, boundary: BoundingBox):
        collision_points_elems = self.collision_points.get_all_cells()
        points = collision_points_elems + [Cell(point[0], point[1], CellLabel.COLLISION) for point in pointcloud]
        
        collision_points = self._generate_new_quadtree(
            self.collision_points.get_boundary().get_bounding_box(), 
            boundary.get_bounding_box(), 
            self.collision_points.get_resolution()
        )

        for point in points:
            collision_points.insert(point)
        
        min_x = collision_points.get_boundary().get_bounding_box()[0][0] if self.boundary.get_bounding_box()[0][0] > collision_points.get_boundary().get_bounding_box()[0][0] else self.boundary.get_bounding_box()[0][0]
        min_y = collision_points.get_boundary().get_bounding_box()[0][1] if self.boundary.get_bounding_box()[0][1] > collision_points.get_boundary().get_bounding_box()[0][1] else self.boundary.get_bounding_box()[0][1]
        
        max_x = collision_points.get_boundary().get_bounding_box()[1][0] if self.boundary.get_bounding_box()[1][0] < collision_points.get_boundary().get_bounding_box()[1][0] else self.boundary.get_bounding_box()[1][0]
        max_y = collision_points.get_boundary().get_bounding_box()[1][1] if self.boundary.get_bounding_box()[1][1] < collision_points.get_boundary().get_bounding_box()[1][1] else self.boundary.get_bounding_box()[1][1]

        self.boundary = BoundingBox((min_x, min_y), (max_x, max_y))
        self.collision_points = collision_points

    def update_covered_points(self, pointcloud, boundary: BoundingBox):
        covered_points_elems = self.covered_points.get_all_cells()
        points = covered_points_elems + [Cell(point[0], point[1], CellLabel.COVERED) for point in pointcloud]
        
        covered_points = self._generate_new_quadtree(
            self.covered_points.get_boundary().get_bounding_box(), 
            boundary.get_bounding_box(), 
            self.covered_points.get_resolution()
        )

        for point in points:
            covered_points.insert(point)
        
        min_x = covered_points.get_boundary().get_bounding_box()[0][0] if self.boundary.get_bounding_box()[0][0] > covered_points.get_boundary().get_bounding_box()[0][0] else self.boundary.get_bounding_box()[0][0]
        min_y = covered_points.get_boundary().get_bounding_box()[0][1] if self.boundary.get_bounding_box()[0][1] > covered_points.get_boundary().get_bounding_box()[0][1] else self.boundary.get_bounding_box()[0][1]
        
        max_x = covered_points.get_boundary().get_bounding_box()[1][0] if self.boundary.get_bounding_box()[1][0] < covered_points.get_boundary().get_bounding_box()[1][0] else self.boundary.get_bounding_box()[1][0]
        max_y = covered_points.get_boundary().get_bounding_box()[1][1] if self.boundary.get_bounding_box()[1][1] < covered_points.get_boundary().get_bounding_box()[1][1] else self.boundary.get_bounding_box()[1][1]

        self.boundary = BoundingBox((min_x, min_y), (max_x, max_y))

        self.covered_points = covered_points

    def update_unexplored_points(self, pointcloud, boundary: BoundingBox):
        unexplored_points_elems = self.unexplored_points.get_all_cells()
        points = [Cell(point[0], point[1], CellLabel.UNEXPLORED) for point in pointcloud] + unexplored_points_elems
        
        unexplored_points = self._generate_new_quadtree(
            self.unexplored_points.get_boundary().get_bounding_box(), 
            boundary.get_bounding_box(), 
            self.unexplored_points.get_resolution()
        )

        for point in points:
            unexplored_points.insert(point)

        min_x = unexplored_points.get_boundary().get_bounding_box()[0][0] if self.boundary.get_bounding_box()[0][0] > unexplored_points.get_boundary().get_bounding_box()[0][0] else self.boundary.get_bounding_box()[0][0]
        min_y = unexplored_points.get_boundary().get_bounding_box()[0][1] if self.boundary.get_bounding_box()[0][1] > unexplored_points.get_boundary().get_bounding_box()[0][1] else self.boundary.get_bounding_box()[0][1]
        
        max_x = unexplored_points.get_boundary().get_bounding_box()[1][0] if self.boundary.get_bounding_box()[1][0] < unexplored_points.get_boundary().get_bounding_box()[1][0] else self.boundary.get_bounding_box()[1][0]
        max_y = unexplored_points.get_boundary().get_bounding_box()[1][1] if self.boundary.get_bounding_box()[1][1] < unexplored_points.get_boundary().get_bounding_box()[1][1] else self.boundary.get_bounding_box()[1][1]

        self.boundary = BoundingBox((min_x, min_y), (max_x, max_y))
        
        self.unexplored_points = unexplored_points

    def generate_path(self, robot_coordinates_x : float, robot_coordinates_y : float):
        self._path_planning = RRT(boundary=self.boundary)

        start = Cell(robot_coordinates_x, robot_coordinates_y, 0)
        goal = self.unexplored_points.find_nearest_neighbour(start)

        path = self._path_planning.generate_path(start, goal, self.collision_points)

        return path
        
