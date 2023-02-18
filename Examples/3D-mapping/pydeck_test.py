import pydeck
import pandas as pd

points = []
with open("point_cloud.xyz", "r") as file:
    for line in file:
        points.append([float(x) for x in line.split()])

df = pd.DataFrame(points, columns=["x","y","z"], dtype = float)

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

r = pydeck.Deck(layers=[point_cloud_layer], initial_view_state=view_state, views=[view], map_provider="mapbox")

r.to_html("env_pydeck.html", css_background_color="#add8e6")