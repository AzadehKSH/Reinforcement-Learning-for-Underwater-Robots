import numpy as np
class BoundingBox:
    def __init__(self, bounding_box_top_left, bounding_box_bottom_right):
        self._bbox = [bounding_box_top_left, bounding_box_bottom_right]
    def get_bounding_box(self):
        return self._bbox
    def contains_cell(self, cell):
        if cell is None:
            return False
        if self._bbox[0][0] <= cell.x <= self._bbox[1][0] and self._bbox[0][1] <= cell.y <= self._bbox[1][1] and self._bbox[0][2] <= cell.z <= self._bbox[1][2]:
            return True
        return False
    def get_middle_point(self):
        return (self._bbox[0][0] + self._bbox[1][0])/2,  (self._bbox[0][1] + self._bbox[1][1])/2, (self._bbox[0][2] + self._bbox[1][2])/2

    def intersects(self, other):
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
        return BoundingBox(
            ( 
                self._bbox[0][1] * np.cos(np.radians(rotation[2])) - np.sin(np.radians(rotation[2]))*self._bbox[0][0], 
                - np.sin(np.radians(rotation[2]))*self._bbox[0][1] + np.cos(np.radians(rotation[2]))*self._bbox[0][0], 
                self._bbox[0][2] 
            ),
            (
                self._bbox[1][1] * np.cos(np.radians(rotation[2])) - np.sin(np.radians(rotation[2]))*self._bbox[1][0], 
                - np.sin(np.radians(rotation[2]))*self._bbox[1][1] + np.cos(np.radians(rotation[2]))*self._bbox[1][0], 
                self._bbox[1][2] 
            )
        )

