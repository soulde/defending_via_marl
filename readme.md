# Multi-USV Defending via MARL
### Author: soulde (李雨璟)
### Github: https://github.com/soulde

# Description
This repository contains the term project of Lesson "Control Theory and Technology of Robot", using MARL algorithm MATD3
to solve the problem of multi-USV defending the landing smuggling boat，NN are trained and eval both in simulate enviroment.
The simulator is part of author's another [open source repo](https://github.com/soulde/soulde_marl_suite). MATD3 is 
compared with the classic algorithm MADDPG, and proved higher efficiency and success rate.

# Installation
Project has been tested under Windows 11 and Ubuntu 22.04 environment.

## Requirements
+ Python 3.9+
+ torch 2.3.0
+ opencv-python 4.9.0

## Direct Installation in Windows or Ubuntu
```shell
cd ${PROJECT_DIR}
pip install -r requirements.txt
pip install -e ./sandbox
```

## Docker build (Untested in Windows)
```shell
cd ${PROJECT_DIR}
docker build -t defending .
```
P.S. If you want to use torch with gpus in docker you should remove default cpu-version torch and install the gpu-version in docker. 
Also, Nvidia-docker2 should be installed in host machine. You can use commands below to install it in the host machine.
### Nvidia-docker2 Install
```shell
sudo apt-get install -y curl
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

# Running

## Direct Installation
```shell
# run td3 eval
python eval_td3.py
# train td3 
python train_td3.py

# run ddpg eval
python eval_ddpg.py
# train ddpg
python train_ddpg.py
```


## Docker

### Start container (container opened with a bash terminal)
```shell
docker run -it --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ./run:/root/defending/run --name defending_env --runtime nvidia --gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" defending
```
You should remove params '--gpus', '--runtime', '--env' if you do not use gpu in the host.
### Open extra terminal
```shell
docker exec -it defending_env bash
```
After start terminal, just work as same as in direct install situation.




