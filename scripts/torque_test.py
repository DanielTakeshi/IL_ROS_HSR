import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import IPython





data = pickle.load(open('debug_data/torque.p','rb'))



x_force = []
y_force = []
z_force = []

x_torque = []
y_torque = [] 
z_torque = []

for torque in data:

	x_force.append(torque.wrench.force.x)
	y_force.append(torque.wrench.force.y)
	z_force.append(torque.wrench.force.z)

	x_torque.append(torque.wrench.torque.x)
	z_torque.append(torque.wrench.torque.y)
	y_torque.append(torque.wrench.torque.z)



plt.plot(x_force,label='x_force')
plt.plot(y_force, label = 'y_force')
plt.plot(z_force, label = 'z_force')

plt.plot(x_torque, label = 'x_torque')
plt.plot(y_torque, label = 'y_torque')
plt.plot(z_torque, label = 'z_torque')


plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.show()

