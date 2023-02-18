import numpy as np
class BoundingBox:
    """Data structure that covers a given three dimensional area. It's similar to a cube.
    """
    def __init__(self, bounding_box_top_left, bounding_box_bottom_right):
        """Data structure that covers a given three dimensional area. It's similar to a cube.

        :param bounding_box_top_left: All values are less than or equal to :param bounding_box_bottom_right:.
        :type bounding_box_top_left: Tuple(Float, Float, Float)
        :param bounding_box_bottom_right: All values are greater then or equal to :param bounding_box_top_left:.
        :type bounding_box_bottom_right: Tuple(Float, Float, Float)
        """
        self._bbox = [bounding_box_top_left, bounding_box_bottom_right]
    def get_bounding_box(self):
        """Getter method for the boundary of the bounding box.

        :return: Boundary of the bounding box.
        :rtype: Tuple(Tuple(Float, Float, Float), Tuple(Float, Float, Float))
        """

        return self._bbox
    def contains_cell(self, cell):
        """Checks if a cell is covered by the bounding box.

        :param cell: The mentioned cell.
        :type cell: Cell
        :return: True if the cell is covered, false otherwise.
        :rtype: Bool
        """

        if cell is None:
            return False
        if self._bbox[0][0] <= cell.x <= self._bbox[1][0] and self._bbox[0][1] <= cell.y <= self._bbox[1][1] and self._bbox[0][2] <= cell.z <= self._bbox[1][2]:
            return True
        return False

    def get_middle_point(self):
        """Get middle point of the bounding box.

        :return: Middle point of the bounding box.
        :rtype: Tuple(Float, Float, Float)
        """
        return (self._bbox[0][0] + self._bbox[1][0])/2,  (self._bbox[0][1] + self._bbox[1][1])/2, (self._bbox[0][2] + self._bbox[1][2])/2

    def intersects(self, other):
        """Checks whether both bounding boxes are intersected.

        :param other: Other bounding box.
        :type other: BoundingBox
        :return: True if the bounding boxes are intersected, false otherwise.
        :rtype: Bool
        """
        return (
            (self._bbox[0][0] <= other._bbox[1][0] and self._bbox[1][0] >= other._bbox[0][0]) and
            (self._bbox[0][1] <= other._bbox[1][1] and self._bbox[1][1] >= other._bbox[0][1]) and
            (self._bbox[0][2] <= other._bbox[1][2] and self._bbox[1][2] >= other._bbox[0][2])
        )
    
    def __add__(self, other):
        return BoundingBox(
            (self._bbox[0][0] + other._bbox[0][0], self._bbox[0][1] + other._bbox[0][1], self._bbox[0][2] + other._bbox[0][2]),
            (self._bbox[1][0] + other._bbox[1][0], self._bbox[1][1] + other._bbox[1][1], self._bbox[1][2] + other._bbox[1][2])
        )
    
    def __mul__(self, rotation):
        # problem with front sonar. usage not recommended.
        def switch(bbox, i):
            bbox[0][i], bbox[1][i] = bbox[1][i], bbox[0][i]
            return bbox

        bbox = [
            [ 
                self._bbox[0][1] * np.cos(np.radians(rotation[2])) - np.sin(np.radians(rotation[2]))*self._bbox[0][0], 
                np.sin(np.radians(rotation[2]))*self._bbox[0][1] + np.cos(np.radians(rotation[2]))*self._bbox[0][0], 
                self._bbox[0][2] 
            ],
            [
                self._bbox[1][1] * np.cos(np.radians(rotation[2])) - np.sin(np.radians(rotation[2]))*self._bbox[1][0], 
                np.sin(np.radians(rotation[2]))*self._bbox[1][1] + np.cos(np.radians(rotation[2]))*self._bbox[1][0], 
                self._bbox[1][2] 
            ]
        ]

        if bbox[0][0] > bbox[1][0]:
            bbox = switch(bbox, 0)
        if bbox[0][1] > bbox[1][1]:
            bbox = switch(bbox, 1)
        if bbox[0][2] > bbox[1][2]:
            bbox = switch(bbox, 2)

        return BoundingBox(
            *bbox
        )

    @staticmethod
    def min_bbox(*bboxes):
        """Calculates a bounding box that contains all given bounding boxes.
        
        :param \*bboxes: Bounding boxes to be checked.
        :type \*bboxes: At least one BoundingBox.
        :return: A bounding box that contains all given bounding boxes.
        :rtype: BoundingBox
        """
        min_x = float("inf")
        min_y = float("inf")
        min_z = float("inf")

        max_x = -float("inf")
        max_y = -float("inf")
        max_z = -float("inf")

        for bbox in bboxes:
            min_x = bbox.get_bounding_box()[0][0] if min_x > bbox.get_bounding_box()[0][0] else min_x
            min_y = bbox.get_bounding_box()[0][1] if min_y > bbox.get_bounding_box()[0][1] else min_y
            min_z = bbox.get_bounding_box()[0][2] if min_z > bbox.get_bounding_box()[0][2] else min_z
            
            max_x = bbox.get_bounding_box()[1][0] if max_x < bbox.get_bounding_box()[1][0] else max_x
            max_y = bbox.get_bounding_box()[1][1] if max_y < bbox.get_bounding_box()[1][1] else max_y
            max_z = bbox.get_bounding_box()[1][2] if max_z < bbox.get_bounding_box()[1][2] else max_z
        
        return BoundingBox((min_x, min_y, min_z), (max_x, max_y, max_z))

