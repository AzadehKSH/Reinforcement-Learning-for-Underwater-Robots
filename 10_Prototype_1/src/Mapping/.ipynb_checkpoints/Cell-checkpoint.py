from collections import namedtuple
from enum import Enum


class CellLabel(Enum):
    EMPTY = 1
    COLLISION = 2
    COVERED = 3
    UNEXPLORED = 4


Cell = namedtuple("Cell", ["x", "y", "label"])