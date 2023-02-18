# Source
## Requirements
- GPU
- Linux machine

## Limitations
- only available for linux, windows wasn't tested but supports HoloOcean
- a gpu is mandatory

# Docker
## Requirements
- Docker
- Nvidia-Docker
- Linux host machine
- GPU

## Execution
You firstly need to build the HoloOcean container. For that execute the terminal command
```console
$ docker compose build holoocean
```
in this directory. After the container was built you can run our system using 
```console
$ docker compose up holoocean
```
This opens a jupyter lab environment on your host machine on port 8888. Here you can run the provided jupyter notebooks. Further you can run `.py` files using the inbuilt terminal of jupyter lab. If you want to visualize the map with Open3D please ensure that the host computer supports OpenGL 3 onwards. If you are connected via an SSH connection with X11 forwarding your computer needs to support OpenGL 3+.
The API and User Documentation can build using the API container. You only need to execute 
```console
$ docker compose build api
```
to compile the API container. Afterwards you can build the documentation with the command: 
```console
$ docker compose up api
```


## Limitations
- only available for linux
- a gpu is mandatory
- no X11 forwarding for HoloOcean

# Improvements
- create a mapping and a visualizer package