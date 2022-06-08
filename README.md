# Mechanical Intelligence for Learning Embodied Sensor-Object Relationships

## Overview

This package contains the algorithm from "Mechanical Intelligence for Learning Embodied Sensor-Object Relationships" published in Nature Communications [^1]. In this paper, we develop an active learning method that enables robots to efficiently collect data for learning a predictive sensor model, called a generative model, without requiring domain knowledge, human input, or previously existing data. This approach drives exploration to information-rich regions based on data-driven sensor characteristics rather than a predefined sensor model. We demonstrate the approach for both cases of near-field sensors (electrolocation) and far-field sensors (grayscale and RBG cameras).

Paper Link: _____________

Website Link: <https://sites.google.com/view/mechanicalintelligence>

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

` virtualenv cbf-learning`

Activate virtual environment:

` source cbf-learning/bin/activate`

Finally, install the dependencies. **Note: Regardless of which environment manager you choose, this step must be completed to create environemnt. **

` pip install -r requirements.txt `

### Get Data for Analysis (Optional)
For those who want to look at the data/models that were learned in this works' examples, you can download the dataset from the zenodo link: __________. Data should be extracted into the `data/` folder in the package. Note that the data files are large, so ensure that there is enough space on your computer before loading it in.

## Getting Started

The package contains code for 3 different examples.

The `electrosensing` folder contains code for the electrolocation example. It contains a jupyter notebook for the data analysis and the algorithm code for the example is located in the `src` folder inside.

The `grayscale_sim` folder contains example code for the Franka robot with a grayscale camera at the end effector with two objects (a cube and a sphere) in the environment in the pybullet simulator.

The `rgb_sim` folder contains example code for the Franka robot with a RGB camera at the end effector with a rubber duck in the environment in the pybullet simulator.

### Final Notes

#### References

[^1] Nature Citation

#### Contact

Contact: Ahalya Prabhakar (ahalya.prabhakar AT epfl dot ch)

(c) apr600