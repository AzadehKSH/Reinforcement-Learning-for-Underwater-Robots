import numpy as np
import open3d as o3d
import plotly.graph_objects as go

points = []
with open("point_cloud_octomap.xyz", "r") as file:
    for line in file:
        points.append([float(x) for x in line.split()])

xxx,yyy,zzz = zip(*points)
# fig = go.Figure(data=[go.Scatter3dg(x=xxx, y=yyy, z=zzz,
#                                    mode='markers',
#                                    marker=dict(
#                                        size=2,
#                                        color=zzz,
#                                        colorscale='blackbody'
#                                    ))]
#                )
# fig.show()


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
print(pcd)
print(len(pcd.points))

o3d.visualization.draw_geometries([pcd])
