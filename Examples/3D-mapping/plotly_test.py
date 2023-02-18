import plotly.graph_objects as go
import numpy as np

points = []
with open("point_cloud.xyz", "r") as file:
    for line in file:
        points.append([float(x) for x in line.split()])

x,y,z = zip(*points)
def scatter_plot(x,y,z):
    
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                    mode='markers',
                                    marker=dict(
                                        size=2,
                                        color=z,
                                        colorscale='blackbody'
                                    ))]
                )
    fig.write_html("environment_scatter_plotly.html")

def surface_plot(x,y,z, surface_dist_between_x_y = 0.15):
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

    for x,y,z in points:
        z_surface = transform_point_to_surface(z_surface, x, y, z, min_x, min_y, max_x, max_y, surface_x_count, surface_y_count)

    fig2_surface = go.Figure(data=[go.Surface(z=z_surface, x=x_surface, y=y_surface)])
    fig2_surface.write_html("environment_surface_plotly.html")

scatter_plot(x,y,z)
surface_plot(x,y,z)