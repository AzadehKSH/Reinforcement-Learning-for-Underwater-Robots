class BoundingBox:
    def __init__(self, bounding_box_top_left, bounding_box_bottom_right):
        self._bbox = [bounding_box_top_left, bounding_box_bottom_right]
    def get_bounding_box(self):
        return self._bbox
    def contains_cell(self, cell):
        if cell is None:
            return False
        if self._bbox[0][0] <= cell.x <= self._bbox[1][0] and self._bbox[0][1] <= cell.y <= self._bbox[1][1]:
            return True
        return False
    def get_middle_point(self):
        return (self._bbox[0][0] + self._bbox[1][0])/2,  (self._bbox[0][1] + self._bbox[1][1])/2

    def intersects(self, other):
        return not (
            other.get_bounding_box()[1][0] < self._bbox[0][0] or
            other.get_bounding_box()[0][0] > self._bbox[1][0] or
            other.get_bounding_box()[1][1] < self._bbox[0][1] or
            other.get_bounding_box()[0][1] > self._bbox[1][1]
        )
    
    def __add__(self, other):
        return BoundingBox(
            (self._bbox[0][0] + other._bbox[0][0], self._bbox[0][1] + other._bbox[0][1]),
            (self._bbox[1][0] + other._bbox[1][0], self._bbox[1][1] + other._bbox[1][1])
        )



