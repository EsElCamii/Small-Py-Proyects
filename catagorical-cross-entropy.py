import math
import numpy as np



softmax_outputs = [0.7, 0.1, 0.2]
taget_output = [1,0,0]

# 2nd and 3rd give 0 
loss = -(math.log(softmax_outputs[0]) * taget_output[0]
         + math.log(softmax_outputs[1]) * taget_output[1]
         + math.log(softmax_outputs[2]) * taget_output[2])

print(loss)

#Its the same since the other values are 0
loss = -math.log(softmax_outputs[0])
print(loss)

#As confindence increases, loss decreases   

 

softmax_outputs = np.array([[0.2, 0.1, 0.2], 
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
class_targets = [0,1,1]

print(-np.log(softmax_outputs[[0,1,2], [class_targets]]))