## About
This is a minimal example of active learning ntfields with igibson setup. 

## Setup
1. git clone this repo
2. run `docker build -t antfields:dev .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --rm -it -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --volume="/home/exx/Documents/antfields:/antfields" --gpus all antfields:dev /bin/bash` to start the docker container. Note that you need to change the path to your own path. BVH library is not installed in the docker image and not necessary for this example. But if you want to use it in evaluation, you need to install it in the docker image by "cd /antfields/bvh-distance-queries && pip install -e ."
4. run `python main.py` to start the training. In main.py, you can switch mode. READ_FROM_COOKED_DATA: train model with all data, EXPLORATION: train model with active learning(exploration) in gibson env.