# IMPROVEMENTS: reimplement unexplored points or calculate unexplored points through explored points
# IMPROVEMENTS: Use mesh plot instead of scatter plot for real time plots (plotly seems kind of unefficient for that task)
# IMPROVEMENTS: Stop subprocess after main process finished.

import holoocean
import numpy as np
import matplotlib.pyplot as plt
import operator

import IPython
import ipywidgets as widgets

from Visualizer.Visualizer import Visualizer
from Mapping.EnvironmentMap import EnvironmentMap
from Mapping.BoundingBox import BoundingBox
import plotly.graph_objects as go
from Mapping.Cell import Cell

from multiprocessing import Process, Manager
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import logging

class Agent():
    """Agent that explores the environment.
    """
    def __init__(self, config, headless, debug, env_resolution):
        """Agent that explores the environment.

        :param config: HoloOceans config dictionary.
        :type config: Dict
        :param headless: True if the agent is run headless, false otherwise
        :type headless: Bool
        :param debug: True if the agent runs in debug mode, false otherwise
        :type debug: Bool
        :param env_resolution: Resolution of the octrees.
        :type env_resolution: Tuple(Tuple(Float, Float, Float), Tuple(Float, Float, Float), Tuple(Float, Float, Float))
        """

        self.map: EnvironmentMap = EnvironmentMap(*env_resolution)

        self.config_front = config['agents'][0]['sensors'][-2]["configuration"]
        self.theta_front = np.linspace(-self.config_front["Azimuth"]/2, self.config_front["Azimuth"]/2, self.config_front["AzimuthBins"])*np.pi/180
        self.r_front = np.linspace(self.config_front["RangeMin"], self.config_front['RangeMax'], self.config_front["RangeBins"])
        self.T_front, self.R_front = np.meshgrid(self.theta_front, self.r_front)
        self.z_front = np.zeros_like(self.T_front)

        self.config_depth = config['agents'][0]['sensors'][-1]["configuration"]
        self.theta_depth = np.linspace(-self.config_depth["Azimuth"]/2, self.config_depth["Azimuth"]/2, self.config_depth["AzimuthBins"])*np.pi/180
        self.r_depth = np.linspace(self.config_depth["RangeMin"], self.config_depth['RangeMax'], self.config_depth["RangeBins"])
        self.T_depth, self.R_depth = np.meshgrid(self.theta_depth, self.r_depth)
        self.z_depth = np.zeros_like(self.T_depth)

        self.rob_pos = ()
        self.rob_rot = ()
        self.rob_forw_vec = ()
        self.env = None
        
        self.phi = None
        self.path = []

        self.headless = headless
        self.debug = debug
        self._debug_server = None
        
        
        if self.debug:
            self._ipython = False
            try:
                shell = get_ipython().__class__.__name__ 
                if shell == 'ZMQInteractiveShell':
                    self._ipython = True
                else:
                    self._ipython = False
            except:
                self._ipython = False
                            
            self._figs = {
                "unexplored": self._init_unexplored_debug(),
                "explored": self._init_explored_debug(),
                "collision": self._init_collision_debug() ,
            }
            
            self._shared_list = None
            if not self._ipython:
                global shared_list
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                manager = Manager()
                shared_list = manager.list([None]*3)
                self._shared_list = shared_list
                self._shared_list[0] = self._figs["unexplored"]
                self._shared_list[1] = self._figs["explored"]
                self._shared_list[2] = self._figs["collision"]
                
                app = dash.Dash("Environment Exploration", update_title=None)
                
                app.layout =  html.Div([
                    html.Div([
                        dcc.Graph(
                        id="unexplored",
                        figure=self._figs["unexplored"]
                    )
                    ]),
                    html.Div([
                        dcc.Graph(
                        id="explored",
                        figure=self._figs["explored"], 
                    ), 
                    ]),
                    html.Div([
                        dcc.Graph(
                        id="collision",
                        figure=self._figs["collision"], 
                    ), 
                    ]), 
                    dcc.Interval(
                        id='interval-component',
                        interval=1*10000, 
                        n_intervals=0
                    )
                ])

                @app.callback(
                    Output("unexplored", "figure"),
                    [Input('interval-component', 'n_intervals')],
                    prevent_initial_call=False
                )
                def update_unexplored(intervals):
                    global shared_list
                    return shared_list[0]

                @app.callback(
                    Output("explored", "figure"),
                    [Input('interval-component', 'n_intervals')],
                    prevent_initial_call=False
                )
                def update_explored(intervals):
                    global shared_list
                    return shared_list[1]

                @app.callback(
                    Output("collision", "figure"),
                    [Input('interval-component', 'n_intervals')],
                    prevent_initial_call=False
                )
                def update_collision(intervals):
                    global shared_list
                    return shared_list[2]
                
                def _run_server(app, **kwargs):
                    import sys
                    import os
                    sys.stdout = open(os.devnull, 'w')
                    app.run_server(**kwargs)

                self._debug_server = Process(target=_run_server, args=(app, ), kwargs={'host': '0.0.0.0', 'debug': False})
                self._debug_server.start()
    
    def init_holoocean(self, config):
        """Initialize the HoloOcean simulator.

        :param config: HoloOceans config.
        :type config: Dict
        """

        if self.headless:
            self.env = holoocean.make(scenario_cfg=config, show_viewport=False)
        else:
            self.env = holoocean.make(scenario_cfg=config)
        
        self.env.reset() 
    
    def _realtime_debug(self):
        self._update_unexplored_debug()
        self._update_explored_debug()
        self._update_collision_debug()
        
        if not self._ipython:
            self._shared_list[0] = self._figs["unexplored"]
            self._shared_list[1] = self._figs["explored"]
            self._shared_list[2] = self._figs["collision"]
        
    @staticmethod
    def _init_unexplored_debug():
        fig = go.FigureWidget()
        
        fig.add_scatter3d(
            mode='markers',
            marker=dict(
                size=2,
                color="black"
            ),
            showlegend=True,
            name="unexpl. points"
        )
        fig.add_scatter3d(
            showlegend=True,
            name="target",
            marker=dict(
                size=5,
                color="blue"
            )
        )
        fig.add_mesh3d(
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=1,
            color='#DC143C',
            flatshading = True,
            showlegend=True,
            name="robot"
        )
        
        fig.update_layout(
            title="unexplored points",
            xaxis_title="y",
            yaxis_title="x" #use same system as holoocean
        )
        
        return fig
    
    @staticmethod
    def _init_explored_debug():
        fig = go.FigureWidget()
        
        fig.add_scatter3d(
            mode='markers',
            marker=dict(
                size=2,
                color="black"
            ),
            showlegend=True,
            name="expl. points"
        )
        fig.add_mesh3d(
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=1,
            color='#DC143C',
            flatshading = True,
            showlegend=True,
            name="robot"
        )
        
        fig.update_layout(
            title="explored points",
            xaxis_title="y",
            yaxis_title="x" #use same system as holoocean
        )
        return fig
    
    @staticmethod
    def _init_collision_debug():
        fig = go.FigureWidget()
        fig.add_scatter3d(
            mode='markers',
            marker=dict(
                size=2,
                color="black"
            ),
            showlegend=True,
            name="coll. points"
        )
        fig.add_mesh3d(
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=1,
            color='#DC143C',
            flatshading = True,
            showlegend=True,
            name="robot"
        )
        
        fig.update_layout(
            title="collision points",
            xaxis_title="y",
            yaxis_title="x" #use same system as holoocean
        )
        return fig
        

    def _update_collision_debug(self, rob_width=1, rob_height=1, rob_depth=1):
        bbox = BoundingBox(
            (self.rob_pos[0]-rob_width, self.rob_pos[1]-rob_height, self.rob_pos[2]-rob_depth),
            (self.rob_pos[0]+rob_width, self.rob_pos[1]+rob_height, self.rob_pos[2]+rob_depth),
        ).get_bounding_box()
        
        world_boundary = self.map.boundary.get_bounding_box()
        
        self._figs["collision"].update_layout(
            scene = dict(
                xaxis = dict(range=[world_boundary[0][0], world_boundary[1][0]]),
                yaxis = dict(range=[world_boundary[0][1], world_boundary[1][1]]),
                zaxis = dict(range=[world_boundary[0][2], world_boundary[1][2]]),
            )
        )
        
        if self.map.collision_points.isEmpty():
            with self._figs["collision"].batch_update():
                self._figs["collision"].data[0].x = []
                self._figs["collision"].data[0].y = []
                self._figs["collision"].data[0].z = []

                self._figs["collision"].data[1].x = [bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]]
                self._figs["collision"].data[1].y = [bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]
                self._figs["collision"].data[1].z = [bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]]
        else:
            with self._figs["collision"].batch_update():
                x,y,z = zip(*[(cell.x, cell.y, cell.z) for cell in self.map.collision_points.get_all_cells()])
                self._figs["collision"].data[0].x = x
                self._figs["collision"].data[0].y = y
                self._figs["collision"].data[0].z = z

                self._figs["collision"].data[1].x = [bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]]
                self._figs["collision"].data[1].y = [bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]
                self._figs["collision"].data[1].z = [bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]]
        
    def _update_explored_debug(self, rob_width=1, rob_height=1, rob_depth=1):
        bbox = BoundingBox(
                    (self.rob_pos[0]-rob_width, self.rob_pos[1]-rob_height, self.rob_pos[2]-rob_depth),
                    (self.rob_pos[0]+rob_width, self.rob_pos[1]+rob_height, self.rob_pos[2]+rob_depth),
        ).get_bounding_box()
        
        world_boundary = self.map.boundary.get_bounding_box()
        self._figs["explored"].update_layout(
            scene = dict(
                xaxis = dict(range=[world_boundary[0][0], world_boundary[1][0]]),
                yaxis = dict(range=[world_boundary[0][1], world_boundary[1][1]]),
                zaxis = dict(range=[world_boundary[0][2], world_boundary[1][2]]),
            )
        )
        
        if self.map.covered_points.isEmpty():
            with self._figs["explored"].batch_update():
                self._figs["explored"].data[0].x = []
                self._figs["explored"].data[0].y = []
                self._figs["explored"].data[0].z = []

                self._figs["explored"].data[1].x = [bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]]
                self._figs["explored"].data[1].y = [bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]
                self._figs["explored"].data[1].z = [bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]]
        else:
            with self._figs["explored"].batch_update():
                x,y,z = zip(*[(cell.x, cell.y, cell.z) for cell in self.map.covered_points.get_all_cells()])
                self._figs["explored"].data[0].x = x
                self._figs["explored"].data[0].y = y
                self._figs["explored"].data[0].z = z

                self._figs["explored"].data[1].x = [bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]]
                self._figs["explored"].data[1].y = [bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]
                self._figs["explored"].data[1].z = [bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]]

    def _update_unexplored_debug(self, rob_width=1, rob_height=1, rob_depth=1):
        bbox = BoundingBox(
            (self.rob_pos[0]-rob_width, self.rob_pos[1]-rob_height, self.rob_pos[2]-rob_depth),
            (self.rob_pos[0]+rob_width, self.rob_pos[1]+rob_height, self.rob_pos[2]+rob_depth),
        ).get_bounding_box()
            
        world_boundary = self.map.boundary.get_bounding_box()
        self._figs["unexplored"].update_layout(
            scene = dict(
                xaxis = dict(range=[world_boundary[0][0], world_boundary[1][0]]),
                yaxis = dict(range=[world_boundary[0][1], world_boundary[1][1]]),
                zaxis = dict(range=[world_boundary[0][2], world_boundary[1][2]]),
            )
        )
        
        if self.map.unexplored_points.isEmpty():
            with self._figs["unexplored"].batch_update():
                self._figs["unexplored"].data[0].x = []
                self._figs["unexplored"].data[0].y = []
                self._figs["unexplored"].data[0].z = []

                self._figs["unexplored"].data[1].x = []
                self._figs["unexplored"].data[1].y = []
                self._figs["unexplored"].data[1].z = []

                self._figs["unexplored"].data[2].x = [bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]]
                self._figs["unexplored"].data[2].y = [bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]
                self._figs["unexplored"].data[2].z = [bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]]
        else:
            with self._figs["unexplored"].batch_update():
                x,y,z = zip(*[(cell.x, cell.y, cell.z) for cell in self.map.unexplored_points.get_all_cells()])
                self._figs["unexplored"].data[0].x = x
                self._figs["unexplored"].data[0].y = y
                self._figs["unexplored"].data[0].z = z

                if len(self.path) > 0:
                    self._figs["unexplored"].data[1].x = [self.path[-1][0]]
                    self._figs["unexplored"].data[1].y = [self.path[-1][1]]
                    self._figs["unexplored"].data[1].z = [self.path[-1][2]]

                bbox = BoundingBox(
                    (self.rob_pos[0]-rob_width, self.rob_pos[1]-rob_height, self.rob_pos[2]-rob_depth),
                    (self.rob_pos[0]+rob_width, self.rob_pos[1]+rob_height, self.rob_pos[2]+rob_depth),
                ).get_bounding_box()

                self._figs["unexplored"].data[2].x = [bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0], bbox[0][0], bbox[0][0], bbox[1][0], bbox[1][0]]
                self._figs["unexplored"].data[2].y = [bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1], bbox[0][1], bbox[1][1], bbox[1][1], bbox[0][1]]
                self._figs["unexplored"].data[2].z = [bbox[0][2], bbox[0][2], bbox[0][2], bbox[0][2], bbox[1][2], bbox[1][2], bbox[1][2], bbox[1][2]]
    
    def _get_coordinates_front(self, r, theta):
        return r * np.cos(theta), r * np.sin(theta), 0

    def _get_coordinates_from_sonar_data_front(self, sonar_data):
        occupied = []
        arr_j = []
        for index_1, i in enumerate(sonar_data):
            for index_2, j in enumerate(i):
                if index_2 in arr_j: continue
                if j == 0.: continue
                coords = self._get_coordinates_front(self.R_front[index_1][index_2], self.T_front[index_1][index_2])
                arr_j += [index_2]
                occupied.append(tuple(map(operator.add, coords, self.rob_pos)))
        return occupied

    def _get_coordinates_depth(self, r, theta):
        return 0, r * np.sin(theta), -r * np.cos(theta) 
    
    def _get_coordinates_from_sonar_data_depth(self, sonar_data):
        result = []
        for index_1, i in enumerate(sonar_data):
            if i.any():
                # array has at least one value other than zero
                for index_2, j in enumerate(i):
                    if j == 0.: continue
                    coords = self._get_coordinates_depth(self.R_depth[index_1][index_2], self.T_depth[index_1][index_2])
                    result.append(tuple(map(operator.add, coords, self.rob_pos)))
        return result

    def run(self, duration = -1, simulation_ticks = 5):
        """Start the environment exploration.

        :param duration: Number of iterations, defaults to -1 for infinite iterations.
        :type duration: Int, optional
        :param simulation_ticks: Number of ticks per action execution, defaults to 5
        :type simulation_ticks: Int, optional
        """
        
        front_ak = self.config_front['RangeMax'] / np.cos(np.deg2rad(self.config_front["Azimuth"]/2.))
        boundary_relativ_front : BoundingBox = BoundingBox((-self.config_front['RangeMax'], -self.config_front['RangeMax'], 0), (self.config_front['RangeMax'], self.config_front['RangeMax'], 0)) # x,y. z is fixed
        
        z_tmp = np.cos(self.theta_depth)*self.config_depth['RangeMax']
        z_rel_uncovered = [z_tmp[0], z_tmp[-1]]
        z_rel_covered = z_tmp[1:-1]
        del z_tmp

        y_tmp = np.sin(self.theta_depth)*self.config_depth['RangeMax']
        y_rel_covered = y_tmp[1:-1]
        y_rel_uncovered = [y_tmp[0], y_tmp[-1]]
        del y_tmp

        x_rel_covered = np.sin(np.deg2rad(self.config_depth["Elevation"]))*self.config_depth['RangeMax']
        depth_gk = z_rel_uncovered[0]
        depth_ak = y_rel_uncovered[-1]
        delta = 0.01
        boundary_relativ_depth : BoundingBox = BoundingBox((-x_rel_covered - delta, -depth_ak - delta, -self.config_depth['RangeMax'] - delta), (x_rel_covered + delta, depth_ak + delta, -self.config_depth['RangeMin'] + delta))

        action = np.zeros(8)
        correct_theta = True
        correct_z = False
        
        self.env_occ = []
        self.env_cov = []
        self.path = []
        self._act(simulation_ticks, action, boundary_relativ_front, boundary_relativ_depth, z_rel_uncovered, y_rel_covered, y_rel_uncovered, 0)
        if self.debug and self._ipython:          
            display(self._figs["explored"])
            display(self._figs["unexplored"])
            display(self._figs["collision"])

        i = 0
        while True:
            i += 1
            
            if self.path is None:
                break
            
            if i == duration:
                break
            if len(self.path) == 0:
                self.path = self.map.generate_path(*self.rob_pos)
                
                if self.path is None or len(self.path) == 0:
                    break
            
            bbox = BoundingBox(
                (self.rob_pos[0] - self.map.covered_points.get_resolution()[0], self.rob_pos[1] - self.map.covered_points.get_resolution()[1], self.rob_pos[2] - self.map.covered_points.get_resolution()[2]),
                (self.rob_pos[0] + self.map.covered_points.get_resolution()[0], self.rob_pos[1] + self.map.covered_points.get_resolution()[1], self.rob_pos[2] + self.map.covered_points.get_resolution()[2])
            )
            for point in reversed(self.path):
                if bbox.contains_cell(Cell(*point, 0)):
                    self.path = self.path[:-1]
                    self.phi = None
                else:
                    break
            
            if len(self.path) == 0:
                continue

            self.env.draw_line(self.rob_pos,self.path[-1])
            self.env.draw_line(self.rob_pos,list(np.add(self.rob_pos, np.multiply(self.rob_forw_vec, 10))))
            relativ_target = np.subtract(self.path[-1],self.rob_pos)
            phi = self._get_phi(self.rob_forw_vec,relativ_target)
            
            if self.phi is None:
                self.phi = phi
            dist = np.linalg.norm(relativ_target)
            
            action = np.zeros(8)
            
            action += self._go_to_phi(phi)[0]
            action += self._go_to_depth(self.path[-1][2])[0]
            action += self._go_to_coordinate(dist)
            
            self._act(simulation_ticks, action, boundary_relativ_front, boundary_relativ_depth, z_rel_uncovered, y_rel_covered, y_rel_uncovered, i)
        # finished

        if self._debug_server is not None:
            self._debug_server.kill()
            
    def _get_phi(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        cross = np.cross(vec1, vec2)
        return np.arctan2(np.dot(cross, [0,0,1]),dot)
    
    def _go_to_phi(self, phi: float):
        action = np.zeros(8)
        
        #print(f"TARGET POINT: {self.path[-1]}, ANGLE: target: {phi*180/np.pi}")
        action[[4,7]] += (phi)*5
        action[[5,6]] -= (phi)*5
        
        return action, 0 if abs(phi) < 0.05 else 1
    
    def _go_to_depth(self, z: float):
        action = np.zeros(8)
        d_z = z - self.rob_pos[2]
        action[0:4] += (z - self.rob_pos[2])*10
        return action, 1 if d_z < self.map.covered_points.get_resolution()[2] else 0
    
    def _go_to_coordinate(self, distance: float):
        action = np.zeros(8)
        action[4:8] += distance
        
        return action
    
    def _act(self, num_ticks, action, boundary_relativ_front, boundary_relativ_depth, z_rel_uncovered, y_rel_covered, y_rel_uncovered, iteration_number):
        self.env.act('auv0', action)
        states = self.env.tick(num_ticks=num_ticks)
        print("\rIteration {}".format(iteration_number + 1), end="")
            
        boundary_robot : BoundingBox = BoundingBox((states["PoseSensor"][0][3], states["PoseSensor"][1][3], states["PoseSensor"][2][3] ),(states["PoseSensor"][0][3], states["PoseSensor"][1][3], states["PoseSensor"][2][3]))
        self.rob_pos = (states["PoseSensor"][0][3], states["PoseSensor"][1][3], states["PoseSensor"][2][3])
        self.rob_rot = states["RotationSensor"]
        self.rob_forw_vec = (states["PoseSensor"][0][0],states["PoseSensor"][1][0],states["PoseSensor"][2][0])
        if 'depth' in states:
            s = states['depth']
            coords = self._get_coordinates_from_sonar_data_depth(s)
            explored_points = [(states["PoseSensor"][0][3], hyp*np.sin(theta), -hyp*np.cos(theta) + states["PoseSensor"][2][3]) for hyp in np.arange(self.config_depth["RangeMin"], self.config_depth['RangeMax'], 0.5) for theta in self.theta_depth[1:-1]]
            # add rotation
            explored_points = [(np.cos(np.radians(self.rob_rot[2]))*pos[1] - np.sin(np.radians(self.rob_rot[2]))*pos[0], np.sin(np.radians(self.rob_rot[2]))*pos[1] + np.cos(np.radians(self.rob_rot[2]))*pos[0] , pos[2]) for pos in explored_points]

            self.map.update_covered_points(explored_points, (boundary_relativ_depth + boundary_robot)*self.rob_rot)
            
            triangle_depth_side = [(states["PoseSensor"][0][3], hyp*np.sin(theta) + states["PoseSensor"][1][3], -hyp*np.cos(theta) + states["PoseSensor"][2][3]) for hyp in np.arange(self.config_depth["RangeMin"],self.config_depth['RangeMax'], 0.5) for theta in [self.theta_depth[0], self.theta_depth[-1]]]
            triangle_depth_bottom = [(states["PoseSensor"][0][3], y + states["PoseSensor"][1][3], -z_rel_uncovered[0] + states["PoseSensor"][2][3]) for y in np.append(y_rel_covered, y_rel_uncovered)]

            unex_depth = triangle_depth_bottom + triangle_depth_side
            unex_front = [(hyp*np.cos(theta) + states["PoseSensor"][0][3],-hyp*np.sin(theta) + states["PoseSensor"][1][3], + states["PoseSensor"][2][3]) for hyp in np.arange(self.config_depth["RangeMin"], self.config_depth['RangeMax'], 0.5) for theta in self.theta_front[1:-1]]

            all_unexplored_points = [(np.cos(np.radians(self.rob_rot[2]))*pos[1] - np.sin(np.radians(self.rob_rot[2]))*pos[0], np.sin(np.radians(self.rob_rot[2]))*pos[1] + np.cos(np.radians(self.rob_rot[2]))*pos[0] , pos[2]) for pos in unex_depth + unex_front]
            
            unexplored_points = []
            bbox = BoundingBox(
                (self.rob_pos[0] - self.map.covered_points.get_resolution()[0], self.rob_pos[1] - self.map.covered_points.get_resolution()[1], self.rob_pos[2] - self.map.covered_points.get_resolution()[2]),
                (self.rob_pos[0] + self.map.covered_points.get_resolution()[0], self.rob_pos[1] + self.map.covered_points.get_resolution()[1], self.rob_pos[2] + self.map.covered_points.get_resolution()[2])
            )
            for point in all_unexplored_points:
                if not bbox.contains_cell(Cell(*point, 0)):
                    unexplored_points.append(point)
            
            unexplored_relative_bbox = BoundingBox.min_bbox(
                boundary_relativ_front,
                boundary_relativ_depth*self.rob_rot
            )
            
            self.map.update_unexplored_points(unexplored_points, unexplored_relative_bbox + boundary_robot, self.rob_pos)
        if 'front' in states:
            s = states['front']        
            occ = self._get_coordinates_from_sonar_data_front(s)
            if len(occ) != 0:
                occ = [(np.cos(np.radians(self.rob_rot[2]))*pos[1] - np.sin(np.radians(self.rob_rot[2]))*pos[0], np.sin(np.radians(self.rob_rot[2]))*pos[1] + np.cos(np.radians(self.rob_rot[2]))*pos[0] , pos[2]) for pos in occ]
                self.env_occ += occ
                boundary_abs_front : BoundingBox = boundary_relativ_front + boundary_robot
                self.map.update_collision_points(occ, boundary_abs_front)
            
        if 'front' in states and 'depth' in states:
            if len(coords) != 0:
                coords = [(np.cos(np.radians(self.rob_rot[2]))*pos[1] - np.sin(np.radians(self.rob_rot[2]))*pos[0], np.sin(np.radians(self.rob_rot[2]))*pos[1] + np.cos(np.radians(self.rob_rot[2]))*pos[0] , pos[2]) for pos in coords]
#                self.map.update_depth_scan(coords)
                self.map.update_collision_points(coords, (boundary_relativ_depth + boundary_robot)*self.rob_rot)
        if self.debug:
            self._realtime_debug()