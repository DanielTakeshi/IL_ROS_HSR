from il_ros_hsr.tensor import inputdata
from il_ros_hsr.p_pi.safe_corl.features import Features
import numpy as np

features = Features()

data = inputdata.IMData(np.array([]), np.array([]), state_space = features.vgg_extract,precompute= True)

data = inputdata.IMData(np.array([]), np.array([]), state_space = features.vgg_kinematic_extract,precompute= True)