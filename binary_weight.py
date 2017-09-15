from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.utils.data as Data

DOWNLOAD_MNIST = True
BATCH_SIZE = 64
SEED = 1024
EPOCH = 1
# Training settings

is_cuda = torch.cuda.is_available()

if is_cuda :
    print('arg cuda available')
    torch.cuda.manual_seed(SEED)
else : 
    torch.manual_seed(SEED)



train_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = True,                                     			       	# this is training data
    transform = torchvision.transforms.ToTensor(),    				# Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download = DOWNLOAD_MNIST,                        				# download it if you don't have it
)

test_data = torchvision.datasets.MNIST(
    root = './mnist/',
    train = False,                                    			       	# this is training data
    transform = torchvision.transforms.ToTensor(),    				# Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download = DOWNLOAD_MNIST,                        				# download it if you don't have it
)

print('test_loader size' , test_data.test_data.size())
print('train_loader size' , train_data.train_data.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = Data.DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle = True)


class Binary_term(nn.Module):
     def __init__(self):
         super(Binary_term , self).__init__()

     def forward(self, weight):
         One = Variable(torch.ones(weight.size()).cuda() , requires_grad = False)
         self.output = torch.sum(torch.pow((weight - One) , 2) * torch.pow((weight + One) , 2))
         return self.output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1 , 20 , 5 , 1)
        self.conv2 = nn.Conv2d(20 , 50 , 5 , 1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4*4*50 , 500)
        self.fc2 = nn.Linear(500 , 10)        
        self.binary_fc2 = Binary_term()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x) , 2 , 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)) , 2 , 2))
        x = x.view(-1 , 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.dropout(x , training = self.training)
        x = self.fc2(x)
        b_loss = self.binary_fc2(self.fc2.weight)
        return F.softmax(x), b_loss

model = Net()
if is_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch , Alpha):
    model.train()
   
    for batch_idx, (data, target) in enumerate(train_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)        
        optimizer.zero_grad()
        output, binary_term_fc2 = model(data)        
        #print(Alpha*binary_term_fc2) 
        loss1 = F.cross_entropy(output, target) 
        loss2 = binary_term_fc2
        loss = loss1 + Alpha*(loss2)       #adding binarization term to loss function 
        loss.backward()
        optimizer.step()
    print('loss1 = ' , loss1)
    print('loss2 = ' , loss2)  
    w_conv1 = model.conv1.weight
    w_conv2 = model.conv2.weight
    w_fc1 = model.fc1.weight       
    w_fc2 = model.fc2.weight
    
    print('w_fc2 = ' , w_fc2)
  


def test():
    model.eval()
    test_loss = 0
    correct = 0
    print(model.fc2.weight)
    for data, target in test_loader:
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, binary_term_fc2 = model(data)
        test_loss += F.cross_entropy(output, target) # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Average loss:' , test_loss , 'Accuracy: ' , 100. * correct / len(test_loader.dataset))

def binarization_weight(w):
    w_n = -(w < 0).type_as(w)
    w_p = (w > 0).type_as(w)
    w_ = w_p + w_n
    w.data = w_.data
    return w
    
    """for i in range(0 , w.size(0)): 
        for j in range(0 , w.size(1)):
            if w.data[i , j] > 0. :
                w.data[i , j] = 1.
            else :
                w.data[i , j] = -1."""

for epoch in range(1, EPOCH + 1):
    Alpha = 0.000001
    c = 1.002
    train(epoch , Alpha)   
    Alpha_ = Alpha
    Alpha = Alpha * pow(c, epoch)   
    print('output layer binarization : ' , ' Epoch = ' , epoch , '/' , EPOCH ,  'c = ' , c , 'Alpha_ = ' , Alpha_ , 'Alpha = ' , Alpha) 
    test()

model.fc2.weight = binarization_weight(model.fc2.weight)
test()
