

import numpy as np
import matplotlib.pyplot as plt

train_imgs_raw = open("MNIST/train-images-idx3-ubyte", "rb")

print("reading training images...")
#reading first 16 bytes = 4 integers that encode: magic_number, p_number_of_images, p_n_rows, p_n_cols
magic_number_imgs = int.from_bytes(train_imgs_raw.read(4), byteorder='big')
m_imgs = int.from_bytes(train_imgs_raw.read(4), byteorder='big')
rows_px = int.from_bytes(train_imgs_raw.read(4), byteorder='big')
cols_px = int.from_bytes(train_imgs_raw.read(4), byteorder='big')

print("magic_number_imgs", magic_number_imgs)
print("m_imgs", m_imgs)
print("rows_px", rows_px)
print("cols_px", cols_px)

train_imgs_buf = train_imgs_raw.read(rows_px*cols_px*m_imgs) 
train_imgs_data = np.frombuffer(train_imgs_buf, dtype=np.uint8).astype(np.float32)
train_imgs = train_imgs_data.reshape(m_imgs, rows_px*cols_px)
train_imgs = train_imgs/255

print("reading training labels...")
train_labels_raw = open("MNIST/train-labels-idx1-ubyte", "rb")
#reading first 2 bytes that encode: magic_number, m_labels
magic_number_labels = int.from_bytes(train_labels_raw.read(4), byteorder='big')    
m_labels = int.from_bytes(train_labels_raw.read(4), byteorder='big')    
    
print("magic_number_labels", magic_number_labels)
print("m_labels ", m_labels )

train_labels_buf = train_labels_raw.read(m_labels)
train_labels = np.frombuffer(train_labels_buf, dtype=np.uint8).astype(np.float32)


def plot_img(imgs, label_data, num):    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot()
    ax.set_title("image "+str(num)+"; label: "+str(int(label_data[num])))
    plt.imshow(imgs[num].reshape(28,28), cmap="Greys")
num = 50
plot_img(train_imgs, train_labels, num)
plt.savefig("example_picture_"+str(num)+".png")
plt.close()


class Net:
    def activation_function(self, z):    
        #sigmoid
        s = 1/(1+np.exp(-z))
        return s

    def activation_function_derivative(self, z):    
        #sigmoid gradient
        s = self.activation_function(z)*(1 - self.activation_function(z))
        return s
        
    def softmax(self, z):                
        exp = np.exp(z)    
        sums = np.sum(exp, axis=0, keepdims=True)    
        res = exp/sums        
        return res
    
    def __init__(self, topology, batch_size):
        
        self.topology = topology
        self.input_dim = topology[0]        
        self.output_dim = topology[-1]                
        self.layers = [{} for x in range(len(topology))]
                
        for i in range(0, len(topology)):            
            if i == 0: 
                W_connections = 1 #never used
            elif i == 1:
                W_connections = self.input_dim                    
            else:
                W_connections = topology[i-1]
            
            #parameters                   
            self.layers[i]["W"] = np.random.uniform(-1, 1, (topology[i], W_connections))
            self.layers[i]["b"] = np.random.uniform(-1, 1, (topology[i],1))
            
            #activations
            self.layers[i]["Z"] = np.zeros((topology[i], batch_size))            
            self.layers[i]["A"] = np.zeros((topology[i], batch_size)) 
            
            #gradients
            self.layers[i]["dW"] = np.zeros((topology[i], W_connections))
            self.layers[i]["db"] = np.zeros((topology[i], batch_size))            
            self.layers[i]["dZ"] = np.zeros((topology[i], batch_size))
            
            
    def feedforward(self, X):                        
        for i in range(0, len(self.topology)-1): 
            
            if i == 0:
                continue            
            elif i == 1:                
                A = X
            else:
                A = self.layers[i-1]["A"]
                
            self.layers[i]["Z"] = np.dot(self.layers[i]["W"], A) + self.layers[i]["b"]  
            self.layers[i]["A"] = self.activation_function(self.layers[i]["Z"]) 
            
        A = self.layers[-2]["A"]
        self.layers[-1]["Z"] = np.dot(self.layers[-1]["W"], A) + self.layers[-1]["b"]          
        self.layers[-1]["A"] = self.softmax(self.layers[-1]["Z"])#
                
        Y_hat = self.layers[-1]["A"]
        
        return Y_hat
        
    def calculate_loss(self, Y_hat, Y_oh):
        batch_size = Y_oh.shape[1]        
        epsilon=1e-10
        return (- np.sum(Y_oh*np.log(Y_hat+epsilon)))/batch_size

    def backprop(self, Y_hat, Y_oh, lr=0.01):                
        #last_layer                
        self.layers[-1]["dZ"] = Y_hat - Y_oh        
        self.layers[-1]["dW"] = np.dot(self.layers[-1]["dZ"], self.layers[-2]["A"].T)        
        self.layers[-1]["db"] = np.mean(self.layers[-1]["dZ"],axis=1, keepdims=True)
        
        #all other layers
        for i in range(len(self.topology)-2, 0,-1):                         
            self.layers[i]["dZ"] = np.dot(self.layers[i+1]["W"].T, self.layers[i+1]["dZ"])*self.activation_function_derivative(self.layers[i]["Z"])                        
            self.layers[i]["dW"] = np.dot(self.layers[i]["dZ"], self.layers[i-1]["A"].T)                        
            self.layers[i]["db"] = np.mean(self.layers[i]["dZ"],axis=1, keepdims=True)
        
        #weight update
        for i in range(0, len(self.topology)):             
            if i == 0:
                continue                                    
            self.layers[i]["W"] = self.layers[i]["W"] - (lr * self.layers[i]["dW"])
            self.layers[i]["b"] = self.layers[i]["b"] - (lr * self.layers[i]["db"])
    

num_of_imgs = len(train_imgs)            

batch_size = 100
batch_number = num_of_imgs//batch_size

train_imgs = train_imgs.T
batches = []
for i in range(0, batch_number):    
    batch_imgs = train_imgs[:,(i*batch_size):(i*batch_size)+batch_size]
    batch_labels = train_labels[(i*batch_size):(i*batch_size)+batch_size]
    batches.append((batch_imgs, batch_labels))

net1 = Net([784, 500, 250, 10], batch_size) #this architecture worked nicely

print("starting training...")

epochs = 20
losses = []

for epoch in range(0, epochs):
    loss = 0
    correct_count = 0    
    for idx, batch in enumerate(batches): 
        
        X, Y = batch        
        
        Y = np.array(Y, dtype=int)                
        
        Y_hat = net1.feedforward(X)        
                        
        pred = np.argmax(Y_hat, axis = 0)
        
        correct_count += np.sum(pred == Y)
        
        Y_oh = np.zeros((10, batch_size), dtype=int)
        
        Y_oh[Y, np.arange(0, Y.shape[0])] = 1        

        loss += net1.calculate_loss(Y_hat, Y_oh)
        
        net1.backprop(Y_hat, Y_oh, lr=0.01)

    avg_loss = loss/batch_number
    accuracy = correct_count/num_of_imgs
    print("epoch", epoch,"\t avg_loss:", avg_loss.round(3), "\taccuracy: ", accuracy.round(3))    
    losses.append(avg_loss)
accuracy = correct_count/num_of_imgs
print(f"last train set epoch ({epoch}):\taccuracy:", accuracy)    

plt.plot(losses)
plt.title("loss")
plt.ylabel("avg_loss")
plt.xlabel("epoch")
plt.show()
plt.savefig("loss.png", dpi=200)


#EVALUATION ON TEST SET
print("Starting test set evaluation")

test_imgs_raw = open("MNIST/t10k-images-idx3-ubyte", "rb")

print("reading training images...")
magic_number_imgs = int.from_bytes(test_imgs_raw.read(4), byteorder='big') #byteorder='little'
m_imgs = int.from_bytes(test_imgs_raw.read(4), byteorder='big')
rows_px = int.from_bytes(test_imgs_raw.read(4), byteorder='big')
cols_px = int.from_bytes(test_imgs_raw.read(4), byteorder='big')

test_imgs_buf = test_imgs_raw.read(rows_px*cols_px*m_imgs) 
test_imgs_data = np.frombuffer(test_imgs_buf, dtype=np.uint8).astype(np.float32)
test_imgs = test_imgs_data.reshape(m_imgs, rows_px*cols_px)
test_imgs = test_imgs/255

print("reading training labels...")
test_labels_raw = open("MNIST/t10k-labels-idx1-ubyte", "rb")

magic_number_labels = int.from_bytes(test_labels_raw.read(4), byteorder='big')    
m_labels = int.from_bytes(test_labels_raw.read(4), byteorder='big')    
    
test_labels_buf = test_labels_raw.read(m_labels)
test_labels = np.frombuffer(test_labels_buf, dtype=np.uint8).astype(np.float32)

num_of_imgs = len(test_imgs)            

batch_size = 100
batch_number = num_of_imgs//batch_size

test_imgs = test_imgs.T
batches = []
for i in range(0, batch_number):    
    batch_imgs = test_imgs[:,(i*batch_size):(i*batch_size)+batch_size]
    batch_labels = test_labels[(i*batch_size):(i*batch_size)+batch_size]
    batches.append((batch_imgs, batch_labels))

print("running on test set")
correct_count = 0    
for idx, batch in enumerate(batches): 
    
    X, Y = batch            
    Y = np.array(Y, dtype=int)                    
    Y_hat = net1.feedforward(X)                            
    pred = np.argmax(Y_hat, axis = 0)    
    correct_count += np.sum(pred == Y)    
    Y_oh = np.zeros((10, batch_size), dtype=int)    
    Y_oh[Y, np.arange(0, Y.shape[0])] = 1
    
test_set_accuracy = correct_count/num_of_imgs
print("accuracy on test set", test_set_accuracy)    










