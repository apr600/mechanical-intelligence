# Mechanical Intelligence for Learning Embodied Sensor-Object Relationships

## Overview

This package contains the algorithm from "Mechanical Intelligence for Learning Embodied Sensor-Object Relationships" published in Nature Communications [^1]. In this paper, we develop an active learning method that enables robots to efficiently collect data for learning a predictive sensor model, called a generative model, without requiring domain knowledge, human input, or previously existing data. This approach drives exploration to information-rich regions based on data-driven sensor characteristics rather than a predefined sensor model. We demonstrate the approach for both cases of near-field sensors (electrolocation) and far-field sensors (grayscale and RBG cameras).

Paper Link: https://rdcu.be/cRIOq

## Installation

### Install Code

To setup the package, go to the directory where you plan to install the code and type:

    git clone https://github.com/apr600/mechanical-intelligence.git

Go to file directory:

    cd mechanical-intelligence

### Setup Environment

Set up python environment (we recommend using virtualenv), but use whatever compatible environment manager that you have with Python3.

    pip install virtualenv

Create folder for virtual environment:

    virtualenv sensory-learning

Activate virtual environment:

    source sensory-learning/bin/activate
### Setup Dependencies

Finally, install the dependencies. *Regardless of which environment manager you choose, this step must be completed to create environemnt.*

    pip install -r requirements.txt

Package Requirements:
- Python>=3.8
- Numpy
- Scipy
- Matplotlib
- Pytorch (Here, we install pytorch version that uses the GPU with Cuda >11.3. If you need to install a different version (to use CPU or for cuda dependencies, please adjust accordingly.)
- Pybullet (for RGB/Grayscale Franka examples)
- Seaborn (optional, for plotting)


## Getting Started

The package contains code for 3 different examples.

The `electrosense` folder contains code for the electrolocation example. It contains a jupyter notebook for the data analysis and the algorithm code for the example is located in the `src` folder inside.

The `grayscale` folder contains example code for the Franka robot with a grayscale camera at the end effector with two objects (a cube and a sphere) in the environment in the pybullet simulator.

The `rgb` folder contains example code for the Franka robot with a RGB camera at the end effector with a rubber duck in the environment in the pybullet simulator.

### Get Data for Analysis (Optional)
For those who want to look at the data/models in this works' examples to generate plots, you can download the dataset from Zenodo as follows below. Data should be extracted into a `data/` directory to be created in here by: 

    mkdir data
    
Note that the data files are large, so we recommend only loading it onto an external drive. Otherwise, ensure that there is enough space on your computer before loading it in.

Once the `data/` directory is created, datasets can be downloaded into the directory with the following structure. 

For the RGB and Electrosensory examples, download the `electrosense.zip` and `rgb.zip` zip files from the Zenodo DOI: <https://doi.org/10.5281/zenodo.6653162>  and extract the directories into the `data/` directory. 



### Final Notes

#### References

[^1] Prabhakar, A., Murphey, T. Mechanical intelligence for learning embodied sensor-object relationships. Nat Commun 13, 4108 (2022). https://doi.org/10.1038/s41467-022-31795-2

#### Contact

Contact: Ahalya Prabhakar (ahalya.prabhakar AT epfl dot ch)

(c) apr600
