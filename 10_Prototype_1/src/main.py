import pickle
from Agent import Agent

def save_voxels(octree, path):
    """Save voxels from the octree.

    :param octree: An octree.
    :type octree: Octree
    :param path: The path where the voxels will be stored.
    :type path: String
    """
    voxels = octree.get_leafs()
    with open(path, "w") as file:
        for voxel in voxels:
            bbox = voxel.get_boundary().get_bounding_box()
            file.write("%f %f %f %f %f %f\n" % (*bbox[0], *bbox[1]))

def save_map(env_map, path):
    """Save the map of the agent.

    :param env_map: Map of the agent.
    :type env_map: EnvironmentMap
    :param path: The path where the map will be stored.
    :type path: String
    """
    with open(path, "wb") as file:
        pickle.dump(env_map, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_octree(octree, path):
    """Save the octree.

    :param octree: An octree.
    :type octree: Octree
    :param path: The path where the octree will be stored.
    :type path: String
    """
    with open(path, "wb") as file:
        pickle.dump(octree, file, protocol=pickle.HIGHEST_PROTOCOL)

def save_pointcloud(octree, path):
    """Save the pointcloud of an octree.

    :param octree: An octree.
    :type octree: Octree
    :param path: The path where the points will be stored.
    :type path: String
    """
    cells = octree.get_all_cells()
    with open(path, "w") as file:
        for cell in cells:
            file.write("%f %f %f\n" % (cell.x, cell.y, cell.z))

if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    def string_to_bool(string):
        string_upper = string.upper()
        if "TRUE".startswith(string_upper):
            return True
        elif "FALSE".startswith(string_upper):
            return False
        else:
            raise Exception("Only values 'true' and 'false' are allowed!")
            exit(1)


    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="confdir", default="./config.json",
                        help="Path (with file name) of the config. Default: config.json in the current directory.")
    parser.add_argument("--voxel", dest="voxdir", default=None,
                        help="Save path of the voxels. Default: no voxels are saved.")
    parser.add_argument("--pointcloud", dest="pcdir", default=None,
                        help="Save path of the pointcloud. Default: no points are saved.")
    parser.add_argument("-m", "--map", dest="mapdir", default="./map.pickle",
                        help="Save path of the map. Default: map.pickle in the current directory.")
    parser.add_argument("--headless", dest="headless", default="TRUE",
                        help="Runs the program in headless mode. There will be no graphical HoloOcean output. Default: True")
    parser.add_argument("-d", "--debug", dest="debug", default="TRUE",
                        help="Activates debug mode. A live visualisation will be shown on port 8050 of the host computer. Default: True")
    parser.add_argument("--unexplored-resolution", dest="unexpl_res", default="1.",
                        help="Resolution of the unexplored octree.  Default: 1.")
    parser.add_argument("--covered-resolution", dest="expl_res", default="1.",
                        help="Resolution of the explored octree.  Default: 1.")
    parser.add_argument("--collision-resolution", dest="coll_res", default="1.",
                        help="Resolution of the explored octree.  Default: 1.")
    parser.add_argument("--max-iterations", dest="max_iter", default="-1",
                        help="Number of iterations.  Default: -1, infinite iterations")
    args = parser.parse_args()
    
    confdir = str(args.confdir)
    voxdir = str(args.voxdir)
    pcdir = str(args.pcdir)
    mapdir = str(args.mapdir)
    headless = string_to_bool(str(args.headless))
    debug = string_to_bool(str(args.debug))
    unexpl_res = float(str(args.unexpl_res))
    expl_res = float(str(args.expl_res))
    coll_res = float(str(args.coll_res))
    max_iter = int(str(args.max_iter))

    with open("./config.json", 'r') as f:
        cfg = json.load(f)
    
    agent = Agent(
        cfg, 
        headless, 
        debug, 
        (
            (coll_res,coll_res,coll_res),
            (expl_res,expl_res,expl_res),
            (unexpl_res,unexpl_res,unexpl_res)
        )
    )

    agent.init_holoocean(cfg)
    agent.run(max_iter)

    if "NONE".startswith(voxdir.upper()):
        save_voxels(agent.map.collision_points, voxdir)
    
    if "NONE".startswith(pcdir.upper()):
        save_pointcloud(agent.map.collision_points, pcdir)
    
    save_map(agent.map, mapdir)