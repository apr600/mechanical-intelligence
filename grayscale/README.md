# Grayscale Camera Model Learning

### Overview

This directory contains the example for learning the grayscale camera sensory model in a Pybullet simulator, where a intensity camera is attached to the end effector of the Franka Panda robot in an environment with two objects, a cube and a sphere.

### Getting Started

The code for running the example is found in the `src/` folder. To run the active learning algorithm from this work, run:

    python intensity_ergodic_main.py

To run the learning comparison with random exploration (i.e., randomly sample location in the workspace), run:

    python intensity_random_main.py

To run the learning comparison with an information maximizing exploration algorithm), run:

    python intensity_infomax_main.py

Note that in the code, there is a `save` parameter, which will save intermediate models during the learning process if set to True (currently set to `False`). Otherwise, only the final models will be saved. The `model_final.pth` and `optim_final.pth` are the models saved at the end of the exploration. The `model_final_postlearning.pth` and `optim_final_postlearning.pth` are the final models saved after the additional learning iterations, which is what we use to evaluate. the `data_dict.pickle` file contains the object location, trajectory data, actions, and the data collected as a buffer. All these files are saved in the `results` folder, in the `ergodic`,  `random`, and `infomax` folders respectively.

### References
