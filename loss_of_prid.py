# linear regression
# y_pred = x*w
# error = (y_prid - y)^2
import np
import matplotlib.pyplot as plt

w = 1.0
# i got some data about the distance of a rocket with variable is time
x_data=[10,15,20,22.5] #second
y_data=[2278.04,362.78,517.35,602.78]

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

mse_list = []
w_list = []
for w in np.arange(10 , 22.5 , 2.4):
    print("w=", w)
    l_sum = 0.0 
    for x_val , y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val,y_val)
        l_sum +=l
        print("\t", x_val, y_val , y_pred_val , l)

    print("MSE=" , l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list,mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show
plt.savefig("rocket_movement.png")