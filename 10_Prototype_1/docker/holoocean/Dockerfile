from frostlab/holoocean:ocean
user root

run rm /etc/apt/sources.list.d/cuda.list
run rm /etc/apt/sources.list.d/nvidia-ml.list
run apt-get update && apt-get install -y mesa-utils libgl1-mesa-glx glibc-source dbus-x11 packagekit-gtk3-module libcanberra-gtk-module
run apt-get install -y libgomp1
run apt-get install -y wget unzip
run pip3 install --upgrade pip

user ue4
run pip3 install numpy==1.22.3 matplotlib==3.5.2 ipython==8.2.0 jupyterlab==3.3.4 pygame==2.1.2 plotly==5.8.0 scikit-learn==1.1.1 
run pip3 install pandas==1.4.2 pydeck==0.7.1
run pip3 install open3d==0.15.2 open3d-python==0.3.0.0
env PATH="${PATH}:/home/ue4/.local/bin"

run wget -P /home/ue4/.local/share/holoocean/0.5.0/worlds  https://unibox.uni-rostock.de/dl/fi5UMq8UwGaPTd3Hvri3hpoc/NewWorld.dir
run mkdir /home/ue4/.local/share/holoocean/0.5.0/worlds/NewWorld
run unzip /home/ue4/.local/share/holoocean/0.5.0/worlds/NewWorld.dir -d /home/ue4/.local/share/holoocean/0.5.0/worlds/NewWorld
run chmod +x "/home/ue4/.local/share/holoocean/0.5.0/worlds/NewWorld/LinuxNoEditor/Holodeck/Binaries/Linux/Holodeck"
run rm /home/ue4/.local/share/holoocean/0.5.0/worlds/NewWorld.dir

run pip install ipywidgets==7.7.2
run pip install dash
