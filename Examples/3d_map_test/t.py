import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

z = z_data.values

sh_0, sh_1 = z.shape

x, y = np.linspace(0, 1, sh_0) * 25, np.linspace(0, 1, sh_1) * 25

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()