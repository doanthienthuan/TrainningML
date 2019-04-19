# gradient_descent
# w=w+learningrate*gradient(x,y)
import np
import matplotlib.pyplot as plt

w1 = 1.0
w2 = 1.0
def forward(x):
    return w1*x*x+w2*x

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)
    
# y_pred=x^2*w2+x*w1+b
def gradient(x,y): #d(loss)/d(w1)=x
    return x
def gradient2(x,y): #d(loss)/d(w2)=x^2
    return x*x


#Training loop

w1_list = []
y_data = [2.0,3.0,6.0]
x_data = [1.0,2.0,3.0]
w2_list = []
loss_list = []

for epoch in range (100):
    for x_val , y_val in zip(x_data,y_data):
        grad1 = gradient(x_val,y_val)
        grad2 = gradient2(x_val,y_val)
        w1 = w1 - 0.01 * grad1
        w2 = w2 - 0.01 * grad2
        print("\tgrad1: ",x_val,y_val,grad1)
        print("\tgrad2: ",x_val,y_val,grad2)
        l=loss(x_val, y_val)
    print("progress:", epoch, "w1=", w1, "loss= ", l)
    w1_list.append(w1)
    loss_list.append(l)
print("predict(after training)", "4 hours can get ", forward(4))

plt.plot(loss_list)
plt.ylabel("Loss")
plt.show()
