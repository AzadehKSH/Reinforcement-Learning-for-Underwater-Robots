# WIP

from enum import Enum
from Mapping.BoundingBox import BoundingBox
import holoocean
import numpy as np

class TestLabel(Enum):
    """Possible test methods of our system.

    :param VOXEL: Voxel test method.
    :type VOXEL: Int, 1
    """
    VOXEL = 1

class Test:
    """Runs test over our system.
    """
    def __init__(self, bboxes, holoocean_world):
        """Runs test over our system.

        :param bboxes: List of the collision octree's bounding boxes.
        :type bboxes: List(BoundingBox)
        :param holoocean_world: World name of the HoloOcean environment.
        :type holoocean_world: String
        """
        self.voxels = bboxes 
        self.world = holoocean_world
        self.env = None

        self._init_holoocean()

    def _init_holoocean(self):
        self.env = holoocean.make(self.world)

    def run_test(self, test_label: TestLabel):
        """Runs tests on our system.

        :param test_label: Test method
        :type test_label: TestLabel
        """
        if test_label == TestLabel.VOXEL:
            self._voxel_test()

    def _voxel_test(self):
        print("See https://holoocean.readthedocs.io/en/master/usage/hotkeys.html for traveling through the environment.")
        def _draw_voxel(voxel: BoundingBox):
            middle = voxel.get_middle_point()
            extent = (
                voxel.get_bounding_box()[1][0] - middle[0],
                voxel.get_bounding_box()[1][1] - middle[1],
                voxel.get_bounding_box()[1][2] - middle[2],
            )
            
            extent = tuple(np.add(extent, 1))
            
            self.env.draw_box(middle, extent, lifetime=0, thickness=100.)
        
        for voxel in self.voxels[:1000]:
            _draw_voxel(voxel)
        
        # run holoocean simulator
        while True:
            self.env.tick(num_ticks=1)

if __name__ == "__main__":
    from argparse import ArgumentParser
    from Visualizer.Visualizer import Visualizer

    def string_to_label(string):
        string_upper = string.upper()
        if "VOXEL".startswith(string_upper):
            return TestLabel.VOXEL
        else:
            raise Exception("Only values 'true' and 'false' are allowed!")
            exit(1)

    parser = ArgumentParser()
    parser.add_argument("-t", "--test-mode", dest="test_mode", default="Voxel", help="Choses the test mode of our system. Available modes: VOXEL")
    parser.add_argument("--voxel", dest="voxdir", default=None,
                        help="Path of the voxels.")
    parser.add_argument("-m", "--map", dest="mapdir", default="./map.pickle",
                        help="Path of the map. Default: map.pickle in the current directory.")
    parser.add_argument("-o", "--octree", dest="octree", default=None,
                        help="Path of the Octree.")
    parser.add_argument("-w", "--world-name", "--world", dest="world_name",
                        help="World name of the HoloOcean environment.")
    args = parser.parse_args()

    test_mode = string_to_label(args.test_mode)
    voxdir = str(args.voxdir) if args.voxdir is not None else None
    mapdir = args.mapdir
    octree_dir = str(args.octree) if args.octree is not None else None
    world_name = str(args.world_name)

    voxels = []

    if voxdir is not None:
        voxels = Visualizer.load_voxel(voxdir)
    elif octree_dir is not None:
        octree = Visualizer.load_octree(octree_dir)
        leafs = octree.get_leafs()

        for leaf in leafs:
            voxels.append(leaf.get_boundary().get_bounding_box())
    else:
        env_map = Visualizer.load_map(mapdir)
        octree = env_map.collision_points
        leafs = octree.get_leafs()

        for leaf in leafs:
            voxels.append(leaf.get_boundary().get_bounding_box())

    if len(voxels) == []:
        exit(1)

    bboxes = []
    for voxel in voxels:
        bboxes.append(BoundingBox(*voxel))

    test = Test(bboxes, world_name)

    if test_mode == TestLabel.VOXEL:
        test.run_test(test_mode)


    

    

