from numpy import less
from Mapping.BoundingBox import BoundingBox
from Mapping.Cell import Cell, CellLabel
from sklearn.neighbors import KDTree

class Octree:
    """Data structure that divides the environment in cubes. Used for efficient storing and querying of three dimensional points.
    """
    def __init__(self, resolution, boundary: BoundingBox, parent=None, depth = 0): 
        """Data structure that divides the environment in cubes. Used for efficient storing and querying of three dimensional points.

        :param resolution: Tuple of three float values. They describe the resolution of the octree on the x, y and z value correspondingly.
        :type resolution: Tuple(Float, Float, Float)
        :param boundary: Bounding box of the octree. The octree only contains a point or children trees in this area.
        :type boundary: BoundingBox
        :param parent: Parent tree, defaults to None
        :type parent: Octree, optional
        :param depth: Number of trees between root and current octree, defaults to 0
        :type depth: int, optional
        """
        self._cell: Cell = None
        self._resolution = resolution
        self._boundary: BoundingBox = boundary
        self._depth = depth
        self._divided = False
        self._children = [None]*8
        self._parent = parent

    def set_cell(self, cell: Cell):
        """Setter method for the cell to store in the octree.

        :param cell: The cell to store.
        :type cell: Cell
        :return: The modified octree.
        :rtype: Octree
        """
        self._cell = cell
        return self
    
    def set_depth(self, depth):
        """Setter method for the depth of an octree.

        :param depth: Number of trees between root and current octree.
        :type depth: int
        :return: The modified octree.
        :rtype: Octree
        """
        self._depth = depth
        return self
    
    def get_resolution(self):
        """Getter method for the resolution of the octree.

        :return: Resolution of the octree.
        :rtype: Tuple(Float, Float, Float)
        """
        return self._resolution
    
    def get_boundary(self):
        """Getter method for the bounding box of the octree.

        :return: Bounding box of the octree. 
        :rtype: BoundingBox
        """
        return self._boundary
    
    def find_cells(self, boundary: BoundingBox):
        """Find cells inside a given bounding box.

        :param boundary: Bounding box that may contain points.
        :type boundary: BoundingBox
        :return: List of cells that are contained in the bounding box.
        :rtype: List(Cell)
        """
        return self._find_cells(boundary, [])
    
    def _find_cells(self, boundary: BoundingBox, found_cells):
        if not self._boundary.intersects(boundary):
            return []

        if self._divided:
            for child in self._children:
                child._find_cells(boundary, found_cells)
        elif boundary.contains_cell(self._cell):
            found_cells.append(self._cell)
        return found_cells
    
    def insert(self, cell: Cell):
        """Inserts a cell into the octree. Divides the octree if there is already a cell and the cell isn't in the near of the cell to be inserted.

        :param cell: The cell to insert.
        :type cell: Cell
        :return: True if cell is inserted, false otherwise
        :rtype: Bool
        """
        if not self._boundary.contains_cell(cell):
            return False
        if self._cell is None and self._children[0] is None:
            self._cell = cell
            return True
        if self._cell is not None:
            if (self._cell.x - self._resolution[0] <= cell.x <= self._cell.x + self._resolution[0] and self._cell.y - self._resolution[1] <= cell.y <= self._cell.y + self._resolution[1] and self._cell.z - self._resolution[2] <= cell.z <= self._cell.z + self._resolution[2]): 
                return True
 
        if not self._divided:
            self._divide()
        
        for child in self._children:
            if child.insert(cell):
                return True
        
        return False
    
    def get_all_cells(self):
        """Get all cells inside an octree.

        :return: List of cells.
        :rtype: List(Cell)
        """
        return self._get_all_cells([])
    
    def get_leafs(self):
        """Get all leaf octrees.

        :return: List of octrees.
        :rtype: list(Octree)
        """
        return self._get_leafs([])

    def _get_all_cells(self, found_cells):
        if self._cell is not None:
            found_cells.append(self._cell)
        if self._divided:
            for child in self._children:
                child._get_all_cells(found_cells)
        return found_cells
    
    def _get_leafs(self, found_cells):
        if self._cell is not None:
            found_cells.append(self)
        if self._divided:
            for child in self._children:
                child._get_leafs(found_cells)
        return found_cells
    
    def _divide(self):
        mid_x, mid_y, mid_z = self._boundary.get_middle_point()
        bbox = self._boundary.get_bounding_box()

        self._children[0] = Octree(
            self._resolution,
            BoundingBox(
              (bbox[0][0], bbox[0][1], bbox[0][2]),
              (mid_x, mid_y, mid_z)
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[1] = Octree(
            self._resolution,
            BoundingBox(
                (mid_x, bbox[0][1], bbox[0][2]),
                (bbox[1][0], mid_y, mid_z)
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[2] = Octree(
            self._resolution,
            BoundingBox(
                (mid_x, mid_y, bbox[0][2]),
                (bbox[1][0], bbox[1][1], mid_z)
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[3] = Octree(
            self._resolution,
            BoundingBox(
                (bbox[0][0], mid_y, bbox[0][2]),
                (mid_x, bbox[1][1], mid_z)
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[4] = Octree(
            self._resolution,
            BoundingBox(
                (bbox[0][0], bbox[0][1], mid_z),
                (mid_x, mid_y, bbox[1][2])
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[5] = Octree(
            self._resolution,
            BoundingBox(
                (mid_x, bbox[0][1], mid_z),
                (bbox[1][0], mid_y, bbox[1][2])
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[6] = Octree(
            self._resolution,
            BoundingBox(
                (mid_x, mid_y, mid_z),
                (bbox[1][0],bbox[1][1],bbox[1][2])
            ),
            parent=self,
            depth=self._depth + 1
        )
        self._children[7] = Octree(
            self._resolution,
            BoundingBox(
                (bbox[0][0], mid_y, mid_z),
                (mid_x, bbox[1][1], bbox[1][2])
            ),
            parent=self,
            depth=self._depth + 1
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
    
    def isEmpty(self):
        """Checks if the octree is empty.

        :return: True if empty, false otherwise
        :rtype: Bool
        """
        return not self._divided and self._cell is None
    
    def get_depth(self):
        """Get depth of the octree.

        :return: Maximal and minimal number of octrees between this octree and the leafs.
        :rtype: Tuple(Int, Int)
        """
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

    def build_kd_tree(self):
        """Build a kd-tree to detect nearest neighbours.
        """
        cells = self.get_all_cells()
        if len(cells) == 0:
            return
        self._kd_tree = KDTree([(cell.x, cell.y, cell.z) for cell in cells])

    def find_nearest_neighbour(self, start_point : Cell):
        """Find nearest neighbour for a given cell.

        :param start_point: The cell whose neighbours are relevant.
        :type start_point: Cell
        :return: Nearest neighbour.
        :rtype: Cell
        """
        _, ind =  self._kd_tree.query([(start_point.x, start_point.y, start_point.z)], k=1)
        point = self._kd_tree.get_arrays()[0][ind]

        return Cell(point[0][0][0], point[0][0][1], point[0][0][2], -1 )

