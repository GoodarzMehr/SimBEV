# SimBEV

![demo](assets/CAM_BACK.gif)

## About

SimBEV a tool that uses the CARLA simulator to generate multi-view images and point clouds that can be used to train computer vision algorithms.

Currently, SimBEV uses the Town06 environment in CARLA. The ego vehicle is equipped with 6 cameras positioned at 0, +-55, +-110, and 180 degree angles (similar to nuScenes) and one 128-beam lidar with a range of 100 meters.
An overhead camera provides the segmented ground truth.
Each camera has a field of view of 80 degrees and outputs a 1600x900 image, which is saved in the .jpg format. The lidar point cloud (x, y, z) is saved as a binary file.
The overhead camera provides a 200x200 image, which covers a 100x100 square around the ego vehicle. Currently, that segmented image is converted into a boolean array for the following four classes: road, car, truck, and pedestrian.

SimBEV randomizes the weather, the spawn point of the ego vehicle, the number, color, and spawn point of other vehicles in the traffic, the number and spawn point of walkers (pedestrians), traffic behavior, and the duration of green lights.

## Installation

### Without Using Docker

1. Install CARLA.
2. In the SimBEV directory, run
```
pip install -r requirements.txt
```

### Using Docker
1. Install Docker on your system.
2. Install the Nvidia Container Toolkit. It exposes your Nvidia graphics card to Docker containers.
3. In the Dockerfile directory, run
```
docker build --no-cache --rm --build-arg ARG -t simbev:develop .
```

## Usage

1. If using Docker, launch a container by running
```
docker run --privileged --gpus all --network=host -e DISPLAY=$DISPLAY
-v [path/to/SimBEV]/simbev:/home/simbev
-v [path/to/dataset]/data:/dataset
--shm-size 32g -it simbev:develop /bin/bash
```
2. Run
```
python simbev.py [options]
```

## Options

* train: number of training scenes (default: 70)
* val: number of validation scenes (default: 15)
* test: number of test scenes (default: 15)
* duration: duration of each scene in seconds (default: 40)
* duration-offset: time passed since the start of the simulation in seconds before data is recorded (default: 5)
* path: path for saving the dataset (default: /dataset)
* render: render sensor data
* no-save: do not save sensor data

### Examples
