import numpy as np
import pickle
import math
import time

from franka.franka_env import FrankaEnv
from franka.franka_utils import *


# Load Pickled Data
# dir_path1 = "results/rgb/entropy1/"
dir_path1 = "results/rgb/entropy1/"
file_path1 = dir_path1 + "data_eval_dict.pickle"
with open(file_path1, 'rb') as f: 
    data_dict1 = pickle.load(f, encoding="bytes")
    print(data_dict1.keys())
env_traj1 = data_dict1['env_path']
env_lim = data_dict1['tray_lim']

env_traj1 = np.array(env_traj1)

# # Initialize Franka Robot
timeStep=1./60.
offset = [0.,0.,0.]
env = FrankaEnv(render=True, ts=timeStep,offset=offset)

### Move robot to starting pose
pos = env_traj1[0]
orn = env.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])
env.step(pos, orn)


input("Press Enter to start trajectory")
for i in range(env_traj1.shape[0]):
    print(i)
    pos = env_traj1[i]
    env.step(pos, orn)
    time.sleep(timeStep*5)

input("Press Enter to end simulation")

