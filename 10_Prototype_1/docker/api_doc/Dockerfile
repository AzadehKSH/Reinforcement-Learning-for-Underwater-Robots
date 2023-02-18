from frostlab/holoocean:ocean
user root

run pip3 install --upgrade pip

user ue4
run pip3 install numpy==1.22.3 matplotlib==3.5.2 pygame==2.1.2 plotly==5.8.0 scikit-learn==1.1.1 
run pip3 install pandas==1.4.2 pydeck==0.7.1
run pip3 install open3d==0.15.2 open3d-python==0.3.0.0
run pip3 install sphinx sphinxcontrib-autoprogram

env PATH="${PATH}:/home/ue4/.local/bin"

workdir /docs
run pip3 install dash
run pip3 install sphinx-typo3-theme
run pip3 install sphinx-copybutton