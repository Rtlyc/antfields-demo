# Start from the iGibson base image or another if more appropriate
FROM igibson/igibson

# Set the working directory (customize as needed)
# WORKDIR /app

# Assuming your requirements.txt is in the same directory as your Dockerfile,
# otherwise, change './requirements.txt' to the correct path inside your project
# COPY ./requirements.txt /app/requirements.txt

# Copy the rest of your application's code
# COPY /home/exx/Documents/igibson-scripts /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    freeglut3-dev \
    libosmesa6-dev \
    patchelf \
    x11-apps

# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch
RUN pip install pickle5
RUN pip install libigl
RUN pip install pytorch-kinematics
RUN pip install --upgrade pyglet==v1.5.28
RUN pip install matplotlib


WORKDIR /antfields
# COPY bvh-distance-queries /antfields/bvh-distance-queries
# RUN cd /antfields/bvh-distance-queries && pip install -e .