import layer
import numpy as np

lay1 = layer.Layer(2,2,True)
lay2 = layer.Layer(2,2,False)

x = np.array([0.05, 0.1])
y = np.array([0.01,0.99])

#forward

out1 = lay1.forward(x)
out2 = lay2.forward(out1)
#print(out1, out2)

grad = -(y - out2)

d_inputs, d_weight, d_bias = lay2.backward(grad)
lay2.update_gd_params(0.5)


"""
for i in range(1):
    out = lay.forward(x)
    print("out: ", out)
    d_inputs, d_weight, d_bias = lay.backward(np.array([[1, 1]]))
    print("Weight grad:", d_weight)
    lay.update_gd_params(1)
    print("Output:", out)
"""
#backward