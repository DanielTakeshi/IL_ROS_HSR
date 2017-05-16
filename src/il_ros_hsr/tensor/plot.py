"""
Visual train and test loss. losses will be plotted as the entire
file which includes all previous optimizations, not just the most recent
"""

import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--net', required=True)
args = vars(ap.parse_args())
path = args['net'] + '/train.log'

f = open(path, 'r')

train_iterations = []
test_iterations = []
train_losses = []
test_losses = []
i = 0
for line in f:
    if '[' in line:
        iter_string = line.split('Iteration ')[1]
        iteration = i + int(iter_string.split(' ]')[0])
        loss = float(iter_string.split('loss: ')[1])
        if "Training" in line:
            train_iterations.append(iteration)
            train_losses.append(loss)
        else:
            test_iterations.append(iteration)
            test_losses.append(loss)
    else:
        i = train_iterations[-1] + 1


plt.plot(train_iterations, train_losses, 'b', label='Training loss')
plt.plot(test_iterations, test_losses, 'r', label='Testing loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()
f.close()
