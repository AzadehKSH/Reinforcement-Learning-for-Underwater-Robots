import sys; sys.path.append("."); sys.path.append("..")
from Mapping.Octree import Octree
import pickle
import plotly.graph_objects as go
import numpy as np
import pydeck
import pandas as pd
import open3d as o3d


class Visualizer():
    def __init__(self, pointcloud = None, voxels = None, octree = None):
        self.pointcloud = pointcloud
        self.voxels = voxels
        self.octree : Octree = octree
        
    def draw_pointcloud_plotly(self):
        x,y,z = zip(*self.pointcloud)
        
        fig = go.Figure(
            data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=z,
                    colorscale='blackbody')
            )]
        )
        return fig
    
    def draw_surface_plotly(self, surface_dist_between_x_y = 0.15):
        x,y,z = zip(*self.pointcloud)
        
        def transform_point_to_surface(z_surface, x, y, z, x_min, y_min, x_max, y_max, x_count, y_count):
            i = int((y - y_min)/(y_max - y_min) * y_count)
            j = int((x - x_min)/(x_max - x_min) * x_count)

            if z_surface[i][j] is None:
                z_surface[i][j] = z

            return z_surface
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)
        max_z = max(z)

        surface_x_count = int((max_x - min_x)/surface_dist_between_x_y)
        surface_y_count = int((max_y - min_y)/surface_dist_between_x_y)

        x_surface, y_surface = np.linspace(min_x, max_x, surface_x_count), np.linspace(min_y, max_y, surface_y_count)

        z_surface = [None] * (surface_y_count + 1)
        for i in range(surface_y_count + 1):
            z_surface[i] = [None] * (surface_x_count + 1)

        for x,y,z in self.pointcloud:
            z_surface = transform_point_to_surface(z_surface, x, y, z, min_x, min_y, max_x, max_y, surface_x_count, surface_y_count)

        fig = go.Figure(data=[go.Surface(z=z_surface, x=x_surface, y=y_surface)])
        
        return fig
    
    def draw_pointcloud_pydeck(self):
        df = pd.DataFrame(self.pointcloud, columns=["x","y","z"], dtype = float)

        point_cloud_layer = pydeck.Layer(
            "PointCloudLayer",
            data=df,
            get_position=["x", "y", "z"],
            get_color=[0,1,1],
            auto_highlight=True,
            pickable=True,
            point_size=3,
            extruded=True,
        )
        target = [df.x.mean(), df.y.mean(), df.z.mean()]
        view = pydeck.View(type="OrbitView", controller=True)
        view_state = pydeck.ViewState(target=target, controller=True, rotation_x=15, rotation_orbit=30, zoom=5.3)

        return pydeck.Deck(layers=[point_cloud_layer], initial_view_state=view_state, views=[view], map_provider="mapbox")
    
    def draw_pointcloud_open3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pointcloud)

        o3d.visualization.draw_geometries([pcd])
        
    def draw_voxel_plotly(self):
        def _draw_vox(fig, bbox, opacity=1):
            fig.add_trace(go.Mesh3d(
            # 8 vertices of a cube
            x=[bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]],
            y=[bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]],
            z=[bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]],
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=opacity,
            color='#DC143C',
            flatshading = True
        ))
            
        fig = go.Figure()
        for voxel in self.voxels:
            _draw_vox(fig, voxel)
        
        return fig
            
    
    @staticmethod
    def load_pointcloud(pcdir):
        points = []
        with open(pcdir, "r") as file:
            for line in file:
                points.append([float(x) for x in line.split()])
                
        return points

    @staticmethod
    def load_voxel(voxdir):
        points = []
        with open(voxdir, "r") as file:
            for line in file:
                num = line.split()
                points.append([(float(num[0]), float(num[1]), float(num[2])), (float(num[3]), float(num[4]), float(num[5]))])
                
        return points
    
    @staticmethod
    def load_octree(ocdir):
        return pickle.load(open(ocdir,"rb"))
    
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-p", "--pointcloud", dest="pcdir",
                        help="Path of the point cloud.")
    parser.add_argument("-v", "--voxel", dest="voxdir",
                        help="Path of the voxels.")
    parser.add_argument("-o", "--octree", dest="ocdir",
                        help="Path of the octree.")
    parser.add_argument("-l", "--library", dest="library",
                        help="Choose the plot engine. Can be either Plotly, Pydeck or Open3D")
    parser.add_argument("-t", "--type", dest="type",
                        help="Choose the type of the plot. Can be either a VOXEL, POINT or SURFACE plot. Surface plots have limitations and are only supported by Plotly. Voxel plots currently are only supported by Plotly")
    parser.add_argument("-s", "--store_html", dest="html",
                        help="Saves the created html file in the given path instead of showing it in the browser. It's only supported by Plotly and Pydeck.")
    args = parser.parse_args()
    
    pointcloud = None; voxels = None; octree = None
    
    print(args.pcdir)
    if args.pcdir is not None:
        pointcloud = Visualizer.load_pointcloud(args.pcdir)
    
    if args.voxdir is not None:
        voxels = Visualizer.load_voxel(args.voxdir)
    
    if args.ocdir is not None:
        octree = Visualizer.load_octree(args.ocdir)

    if args.type is None:
        raise Exception("Add plot type.")
        
    vis = Visualizer(pointcloud, voxels, octree)
    
    if args.type.lower() == "voxel":
        if args.library.lower() == "plotly":
            fig = vis.draw_voxel_plotly()
        else:
            raise Exception("Voxel plots currently are only supported by plotly")
       
    elif args.type.lower() == "point":
        if args.library.lower() == "plotly":
            fig = vis.draw_pointcloud_plotly()
        elif args.library.lower() == "pydeck":
            fig = vis.draw_pointcloud_pydeck()
        elif args.library.lower() == "open3d":
            fig = vis.draw_pointcloud_open3d()
        else:
            raise Exception("Invalid library.")
    
    elif args.type.lower() == "surface": 
        if args.library.lower() == "plotly":
            fig = vis.draw_surface_plotly()
        else: 
            raise Exception("Surface plots are only supported by plotly.")
    
    if args.html is not None:
        if args.library.lower() == "plotly":
            fig.write_html(args.html)
        elif args.library.lower() == "pydeck":
            fig.to_html(args.html, css_background_color="#add8e6")
        else:
            raise Exception("HTML-files can be created only for Plotly and Pydeck.")
    else:
        if args.library.lower() == "plotly":
            fig.show()
        elif args.library.lower() == "pydeck":
            fig.show()