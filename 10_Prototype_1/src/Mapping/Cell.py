from collections import namedtuple
from enum import Enum


class CellLabel(Enum):
    """Label of a cell. A cell can be either empty, a collision point, covered from the depth sonar sensor or unexplored.
    """
    EMPTY = 1
    COLLISION = 2
    COVERED = 3
    UNEXPLORED = 4


Cell = namedtuple("Cell", ["x", "y", "z", "label"])
"""Cell with an x, y and z coordinate as well as a label.

:param x: The x-coordinate of the cell.
:type x: Float

:param y: The y-coordinate of the cell.
:type y: Float

:param z: The z-coordinate of the cell.
:type z: Float

:param label: The label of the cell.
:type label: CellLabel
"""