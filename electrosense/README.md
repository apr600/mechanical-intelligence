# Electrosensing

### Overview

This directory contains the electrosensing example for learning the electrolocation sensory modality used commonly in weakly electric fish found in nature [^1].

### Getting Started

The code for running the example is found in the `src/` folder. To run the active learning algorithm from this work, run:

    python electro_ergodic_main.py

To run the learning comparison with random exploration (i.e., randomly sample location in the workspace), run:

    python electro_random_main.py


Note that in the code, there is a `save` parameter, which will save intermediate models during the learning process if set to True (currently set to `False`). Otherwise, only the final models will be saved. The `model_final.pth` and `optim_final.pth` are the models saved at the end of the exploration. The `model_final_postlearning.pth` and `optim_final_postlearning.pth` are the final models saved after the additional learning iterations, which is what we use to evaluate. the `data_dict.pickle` file contains the object location, trajectory data, actions, and the data collected as a buffer. All these files are saved in the `results` folder, in the `ergodic` and `random` folders respectively.

### References
