from Mapping.BoundingBox import BoundingBox
from Mapping.Cell import Cell
class Quadtree:
    def __init__(self, resolution, boundary: BoundingBox, depth = 0):
        self._cell: Cell = None
        self._resolution = resolution
        self._boundary: BoundingBox = boundary
        self._depth = depth
        self._divided = False
        self._children = [None]*4

    def set_cell(self, cell: Cell):
        self._cell = cell
        return self
    
    def set_depth(self, depth):
        self._depth = depth
        return self
    
    def get_resolution(self):
        return self._resolution
    
    def get_boundary(self):
        return self._boundary
    
    def find_cells(self, boundary: BoundingBox):
        return self._find_cells(boundary, [])
    
    def _find_cells(self, boundary: BoundingBox, found_cells):
        if not self._boundary.intersects(boundary):
            return False

        if self._divided:
            for child in self._children:
                child._find_cells(boundary, found_cells)
        elif boundary.contains_cell(self._cell):
            found_cells.append(self._cell)
        return found_cells
    
    def insert(self, cell: Cell):
        if not self._boundary.contains_cell(cell):
            return False
        if self._cell is None and self._children[0] is None:
            self._cell = cell
            return True
        if self._cell is not None:
            if (self._cell.x - self._resolution[0] <= cell.x <= self._cell.x + self._resolution[0] and self._cell.y - self._resolution[1] <= cell.y <= self._cell.y + self._resolution[1]): 
                return True
 
        if not self._divided:
            self._divide()
        
        for child in self._children:
            if child.insert(cell):
                return True
        
        return False
    
    def get_all_cells(self):
        return self._get_all_cells([])

    def _get_all_cells(self, found_cells):
        if self._cell is not None:
            found_cells.append(self._cell)
        if self._divided:
            for child in self._children:
                child._get_all_cells(found_cells)
        return found_cells
    
    def _divide(self):
        mid_x, mid_y = self._boundary.get_middle_point()
        bbox = self._boundary.get_bounding_box()

        self._children[0] = Quadtree(
            self._resolution, 
            BoundingBox(bbox[0], (mid_x, mid_y)),
            depth = self._depth + 1
        )

        self._children[1] = Quadtree(
            self._resolution, 
            BoundingBox((bbox[0][0], mid_y), (mid_x, bbox[1][1])),
            depth = self._depth + 1
        )

        self._children[2] = Quadtree(
            self._resolution, 
            BoundingBox((mid_x, bbox[0][1]), (bbox[1][0], mid_y)),
            depth = self._depth + 1
        )

        self._children[3] = Quadtree(
            self._resolution, 
            BoundingBox((mid_x, mid_y), bbox[1]),
            depth = self._depth + 1
        )
        
        for child in self._children:
            if child.insert(self._cell):
                break
        self._cell = None

        self._divided = True
    
    def __len__(self):
        length = 0
        if self._cell is not None:
            length = 1
        elif self._divided:
            length = sum(len(child) for child in self._children)

        return length
    def get_depth(self):
        if self._children[0] is None:
            return (self._depth, self._depth)
        else:
            min_depth = self._children[0].get_depth()[0]
            max_depth = self._children[0].get_depth()[1]

        for child in self._children[1:]:
            depth = child.get_depth()
            if depth[0] < min_depth:
                min_depth = depth[0]
            if depth[1] > max_depth:
                max_depth = depth[1]
            return min_depth, max_depth
    
    def find_nearest_neighbour(self, start_point : Cell):
        if not self._boundary.contains_cell(start_point):
            return None
        if self._children[0] is None:
            return self._cell

        for child in self._children:
            nearest_neighbour = child.find_nearest_neighbour(start_point)
            if nearest_neighbour is not None:
                return nearest_neighbour
        
        return None
