import numpy as np
import time
import matplotlib.pyplot as plt



train_images_np=np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np=np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np=np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np=np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np=np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np=np.load('./Project3_Data/MNIST_test_labels.npy')


##Template MLP code
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))

class MLP():

    def __init__(self):
        #Initialize all the parametres
        #Uncomment and complete the following lines
        self.W1= np.random.normal(0, 0.1, size=(784, 64))
        self.b1= np.zeros(shape=(1, 64))
        self.W2= np.random.normal(0, 0.1, size=(64, 10))
        self.b2= np.zeros(shape=(1, 10))
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        #Feed data through the network
        #Uncomment and complete the following lines
        self.x=x
        self.W1x= np.dot(self.x, self.W1)
        self.a1= self.W1x + self.b1
        self.f1= sigmoid(self.a1)
        self.W2x= np.dot(self.f1, self.W2)
        self.a2= self.W2x + self.b2
        self.y_hat= softmax(self.a2)
        return self.y_hat

    def update_grad(self,y):
        # Compute the gradients for the current observation y and add it to the gradient estimate over the entire batch
        # Uncomment and complete the following lines
        dA2db2= 1
        dA2dW2= self.f1
        dA2dF1= self.W2
        dF1dA1= sigmoid(self.a1) * (1-sigmoid(self.a1))
        dA1db1= 1
        dA1dW1= np.expand_dims(self.x, axis=0)

        dLdA2 = self.y_hat - y
        dLdW2 = dA2dW2.T * dLdA2
        dLdb2 = dA2db2 * dLdA2
        dLdF1 = np.dot(dLdA2, dA2dF1.T)
        dLdA1 = dF1dA1 * dLdF1
        dLdW1 = dA1dW1.T * dLdA1
        dLdb1 = dLdA1 * dA1db1

        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1

        pass

    def update_params(self,learning_rate):
        self.W2 = self.W2 - learning_rate * self.W2_grad
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1)
        self.W1 = self.W1 - learning_rate * self.W1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)




def toOneHot(arr):
    b = np.zeros((arr.size, arr.max()+1) ,dtype=np.int8)
    b[np.arange(arr.size),arr] = 1
    return b

def compute_confusion_matrix(true, pred):
  result = np.zeros((10, 10))
  for i in range(len(true)):
    result[np.argmax(true[i])][np.argmax(pred[i])] += 1
  return result



def accuracy(images, labels):
    correct = 0
    total = len(images)
    for i in range(len(images)):
        outputs = myNet.forward(images[i])
        if np.argmax(outputs[0]) == np.argmax(labels[i]):
            correct += 1
    return (100 * correct / total)



## Init the MLP
myNet=MLP()
myNet.reset_grad()

learning_rate=1e-3
n_epochs=100
batch_size = 256

trainAccHist = []
valAccHist = []
lossHist = []

train_labels_np = toOneHot(train_labels_np)
val_labels_np = toOneHot(val_labels_np)
test_labels_np = toOneHot(test_labels_np)

totalImages = len(train_images_np)
#totalImages = 2000

## Training code
for epoch in range(n_epochs):
    #Code to train network goes here
    running_loss = 0.0

    for i in range(0, totalImages):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = train_images_np[i], train_labels_np[i]
        

        # forward + backward + optimize
        outputs = myNet.forward(inputs)
        loss = CrossEntropy(outputs[0], labels)

        myNet.update_grad(labels)
        if i%batch_size == 0:
            myNet.update_params(learning_rate)
            myNet.reset_grad()

        running_loss += loss

    #Code to compute validation loss/accuracy goes here
    trainAcc = accuracy(train_images_np[:totalImages], train_labels_np[:totalImages])
    valAcc = accuracy(val_images_np, val_labels_np)
    finalEpochLoss = running_loss/totalImages

    trainAccHist.append(trainAcc)
    valAccHist.append(valAcc)
    lossHist.append(finalEpochLoss)
    print(f"Epoch {epoch+1} - Train Accuracy {trainAcc:.2f}% - Val Accuracy {valAcc:.2f}% - Loss {finalEpochLoss:.4}")

print(f"Test Accuracy: {accuracy(test_images_np, test_labels_np)}")

plt.figure(1)
plt.plot(trainAccHist)
plt.plot(valAccHist)
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Accuracy vs Epochs")
plt.savefig('./plots/Accuracy_mynn.png')

pred = []
for x in range(len(test_images_np)):
    pred.append(myNet.forward(test_images_np[x]))

matrix = compute_confusion_matrix(test_labels_np, pred)

plt.figure(2)
plt.imshow(matrix, interpolation='nearest')
plt.xticks(np.arange(0,10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.yticks(np.arange(0,10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.xlabel("True")
plt.ylabel("Pred")
plt.title("Confusion Matrix")
plt.savefig('./plots/Confusion Matrix.png')


plt.figure(3)
plt.imshow(myNet.W2)
plt.title("W2")
plt.savefig('./plots/W2.png')

plt.figure(4)
plt.imshow(myNet.W1)
plt.title("W1")
plt.savefig('./plots/W1.png')

plt.figure(5)
plt.imshow(myNet.b2)
plt.title("b2")
plt.savefig('./plots/b2.png')

plt.figure(6)
plt.imshow(myNet.b1)
plt.title("b1")
plt.savefig('./plots/b1.png')

# save the weights and biases
np.save("./weights and biases/w1.npy", myNet.W1)
np.save("./weights and biases/b1.npy", myNet.b1)
np.save("./weights and biases/w2.npy", myNet.W2)
np.save("./weights and biases/b2.npy", myNet.b2)


# Reset the data
train_images_np=np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np=np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np=np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np=np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np=np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np=np.load('./Project3_Data/MNIST_test_labels.npy')

## Template for ConvNet Code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    #From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.view(-1,1,28,28))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Your training and testing code goes here

net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def accuracyCalc(dataImages, dataLabels):
    correct = 0
    total = 0
    with torch.no_grad():
        dataInputs, dataLabels = dataImages, dataLabels
        dataInputs, dataLabels= torch.Tensor(dataInputs), torch.Tensor(dataLabels).long()
        outputs = net(dataInputs)
        _, predicted = torch.max(outputs.data, 1)
        total += dataLabels.size(0)
        correct += (predicted == dataLabels).sum().item()
        return (100 * correct / total)


trainAccHist = []
valAccHist = []
lossHist = []
for epoch in range(n_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0, totalImages, batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = train_images_np[i:i+batch_size], train_labels_np[i:i+batch_size]
        inputs = torch.Tensor(inputs)
        
        labels = torch.Tensor(labels).long()
        # labels = labels.reshape(-1,1)
        # labels = torch.nn.functional.one_hot(labels.long(), num_classes=10)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    trainAcc = accuracyCalc(train_images_np[:totalImages], train_labels_np[:totalImages])
    valAcc = accuracyCalc(val_images_np, val_labels_np)

    trainAccHist.append(trainAcc)
    valAccHist.append(valAcc)
    lossHist.append(running_loss)
    
    print(f'Epoch {epoch + 1} Train Accuracy {trainAcc}, Val Accuracy {valAcc}, loss {running_loss}')


print(f'Test Accuracy {accuracyCalc(test_images_np, test_labels_np)}')





plt.figure(7)
plt.plot(trainAccHist)
plt.plot(valAccHist)
plt.xlabel("Epochs")
plt.ylabel("Accuracy %")
plt.title("Pytorch Accuracy vs Epochs")
plt.savefig('./plots/Pytorch accuracy.png')

plt.figure(8)
plt.plot(lossHist)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Pytorch Loss vs Epochs")
plt.savefig('./plots/Pytorch loss.png')



torch.save(net.state_dict(), "./weights and biases/pytorch.pth")

print('Finished Training')

