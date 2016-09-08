import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print(sys.path)

import numpy as np
import neuron
import activation_function

no_inputs = 2
af = activation_function.Build("sigmoid")
rho = 1

# standalone neutron for arithmatic addition
sn_add = neuron.StandaloneNeuron(no_inputs, af, rho, True)
for i in range(500):
    input = np.random.uniform(0, 1, 2)
    output = sum(input)/2
    #print("input = {}, output = {}".format(input, output))
    sn_add.Train(input, output)

print(sn_add.GetOutput(np.array([0.1,0.1]))*2)
print(sn_add.GetOutput(np.array([0.2,0.1]))*2)
print("the trained weights: {}, {}".format(sn_add._weight, sn_add._bias))
