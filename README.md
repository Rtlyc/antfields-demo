## About
This is a minimal example of active learning ntfields with igibson setup. 

## Setup
1. git clone this repo
2. run `docker build -t antfields:demo .` under the root directory of this repo, once you built the docker image, you don't need to build it again unless you change the dockerfile.
3. run `docker run --rm -it -e DISPLAY=$DISPLAY --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --volume="/home/exx/Documents/antfields-demo:/antfields" --gpus all antfields:demo /bin/bash` to start the docker container. Note that you need to change the path to your own path. BVH library is only needed during evaluation.
4. run `python main.py` to start the training. In main.py, you can switch mode. READ_FROM_COOKED_DATA: train model with previous collected data, EXPLORATION: train model with active learning(exploration) in gibson env. 
```
# train model with default EXPLORATION mode
python main.py 

# train model with READ_FROM_COOKED_DATA mode
python main.py --no_explore

# evaluate model
python eval.py
```

## Note

We use bvh-distance-queries library [repo](https://github.com/YuliangXiu/bvh-distance-queries?tab=readme-ov-file) to calculate the distance between two objects. If the bvh library fails to install in dockerfile, you can install it manually by "cd /antfields/bvh-distance-queries && pip install -e .".
Occupancy map is modified from [repo](https://github.com/richardos/occupancy-grid-a-star) to fit the igibson env.
Data sampling and preprocessing is modified from [repo](https://github.com/facebookresearch/iSDF)
