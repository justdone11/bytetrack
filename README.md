# Byte Track Wrapper based on YoloX


## Installation

Please, install the requirements, including pytorch. Tested with python=3.9

Please use numpy < 1.24

```
pip install lap cython scipy opencv-python
pip install cython_bbox
```

If using windows: `pip install git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox`


## Pretrained

```
pip install gdown
gdown -O pretrained/bytetrack_x_mot17.pth.tar 1HsIYBFmPUWGLHjSKSSeYXcUELfFnN1SX
```

### tensorRT acceleration [Optional]

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

### CAUTION HERE [OPTIONAL]

In case tensorRT plugins are required (very optional):

 `cmake -B build . && cmake --build build --target install && ldconfig`

Furthermore, experimental community contribution (requires tensorRT >= 7.0) can be installed with: 
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt/scripts    
bash build_contrib.sh   
```
