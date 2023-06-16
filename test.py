import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential 
from tensorflow.python.keras.layers import Dense

# x = Dense(units=1,input_shape = [1])
# birds = [
# 		["Barn Owl", "Owl", "Green"]
# 		["Black tern", "Gulls and terns", "Green"]
#         ["Barn Owl", "Owl", "G2331reen"]
#         ["Barn Owl", "Owl", "G322reen"]
#         ["Barn Owl", "Owl", "Green"]
#         ["Barn Owl", "Owl", "G3123reen"]
#         ["Barn Owl", "Owl", "Gr3213een"]
#         ["Barn Owl", "Owl", "Green"]
#         ["Barn Owl", "Owl", "Gre321en"]
#         ["Barn Owl", "Owl", "Green"]
#         ["Barn Owl", "Owl", "Gr32een"]
#         ["Barn Owl", "Owl", "Greeewn"]
#         ["Barn Owl", "Owl", "Grddeen"]
#         ["Barn Owl", "Owl", "Grdween"]
#         ["Barn Owl", "Owl", "Greend"]
#         ["Barn Owl", "Owl", "Greedn"]
#         ["Barn Owl", "Owl", "Greden"]
#         ["Barn Owl", "Owl", "Gddreen"]

# ]
v = [1,2,3,4,5,6,7,8,9,0]

index =3 
numbers=[1,2,3,4]
print(v[-index:,:])

# X_test = birds[-index:, :]

# print(X_test)
# model = Sequential([x])

# model.compile(optimizer='sgd',loss='mean_squared_error')

# xs= np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)

# ys= np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)

# model.fit(xs,ys,epochs=100)

# print(model.predict([4]))

# print(x.get_weights())