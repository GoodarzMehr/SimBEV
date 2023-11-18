# SimBEV

![demo](assets/MultiView.gif)

## About

SimBEV is a data generation tool that leverages the [CARLA Simulator](https://github.com/carla-simulator/carla) to record multi-view images and point clouds from randomized scenarios that can be used to train and evaluate autonomous vehicle perception algorithms, especially [BEVFusion](https://github.com/mit-han-lab/bevfusion). The user can specify the number of training, validation, and testing scenarios, as well as the duration of each scenario.

The ego vehicle is currently equipped with six cameras and one lidar. The cameras are positioned at the $0$, $\pm55$, $\pm110$, and $180$ degree angles, similar to the [nuScenes dataset](https://www.nuscenes.org/). Each camera has a resolution of $1600\times900$ and an $80$-degree field of view, and the images are saved in the `*.jpg` format. The lidar has $128$ beams and a range of $100$ meters, and its data ($x$, $y$, and $z$ coordinates) is saved as a binary file. Finally, an overhead semantic segmentation camera provides the ground truth. It covers a $100\times100$-meter square centered on the ego vehicle with a resolution of $0.5$ meters, outputting a $200\times200$ image. The segmented image is currently converted into a boolean array and saved as a binary file for the following four classes: road, car, truck, and pedestrian.

For each scenario, SimBEV randomizes the weather; the starting point of the ego vehicle; the number, color, and starting point of other vehicles in the traffic; the number and starting point of walkers (pedestrians); traffic behavior; and the duration of green lights.

The recorded data is saved into three folders: `sweeps`, which contains the images and point cloud binary files, `ground-truth`, which contains the ground truth binary files, and `infos`, which contains `*.json` files that provide information about the data.

## Installation

### Without Using Docker

1. Install the [CARLA Simulator](https://github.com/carla-simulator/carla).
2. In the SimBEV directory, run
```
pip install -r requirements.txt
```

### Using Docker

1. Install [Docker](https://docs.docker.com/engine/install/) on your system.
2. Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide). It exposes your Nvidia graphics card to Docker containers.
3. In the SimBEV directory, run
```
docker build --no-cache --rm -t simbev:develop .
```

The use of the following build arguments is optional:
* `USER`: default username inside each container, set to `sb` by default.
* `CARLA_VERSION`: desired version of CARLA, set to `0.9.14` by default.

## Usage

If you are using Docker, launch a container using
```
docker run --privileged --gpus all --network=host -e DISPLAY=$DISPLAY
-v [path/to/SimBEV]/simbev:/home/simbev
-v [path/to/dataset]/data:/dataset
--shm-size 32g -it simbev:develop /bin/bash
```
Run
```
python simbev.py [options]
```

## Options

* `train`: number of training scenes (default: $70$)
* `val`: number of validation scenes (default: $15$)
* `test`: number of test scenes (default: $15$)
* `duration`: duration of data recording in each scene in seconds (default: $40$)
* `duration-offset`: time passed since the start of the simulation in seconds before data is recorded (default: $5$)
* `path`: path for saving the dataset (default: `/dataset`)
* `render`: also render sensor data
* `no-save`: do not save sensor data

## Examples

The following example only renders sensor data and does not save anything.
```
python simbev.py --no-save --render
```

The following example both saves and renders sensor data for $10$ training, $7$ validation, and $3$ test scenarios. In each scenario $20$ seconds of data is recorded after the first $7$ seconds have passed. Data is saved to the `/some/data/path` path.
```
python simbev.py --render --train 10 --val 7 --test 3 --duration 20 --duration-offset 7 --path '/some/data/path'
```
