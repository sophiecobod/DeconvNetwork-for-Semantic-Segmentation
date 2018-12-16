import torch
import torch.nn as nn 
import os 
import numpy as np
import torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from model import conv_deconv 

class ImageDataset(Dataset):
  
  
  
train_dataset=ImageDataset(input_dir="input",output_dir="output") #Training Dataset
test_dataset=ImageDataset(input_dir="input",output_dir="output",train=False) #Testing Dataset
batch_size = 10 #mini-batch size
n_iters = 10000 #total iterations
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
  
 
model = conv_deconv()

iter=0
iter_new=0 
check=os.listdir("checkpoints") #checking if checkpoints exist to resume training
if len(check):
    check.sort(key=lambda x:int((x.split('_')[2]).split('.')[0]))
    model=torch.load("checkpoints/"+check[-1])
    iter=int(re.findall(r'\d+',check[-1])[0])
    iter_new=iter
    print("Resuming from iteration " + str(iter))

if torch.cuda.is_available(): #use gpu if available
	model.cuda()

def train_conv_deconv(model,size,conv_feat=None,labels=None,epochs=1,optimizer=None,train=True,shuffle=True):
    if train:
        model.train()
    else:
        model.eval()
        
criterion = nn.MSELoss() #MSquaredLOSS
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

beg=time.time() #time at the beginning of training
print("Training Started!")
for epoch in range(num_epochs):
    print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
    for i,datapoint in enumerate(train_loader):
        datapoint['input_image']=datapoint['input_image'].type(torch.FloatTensor) #typecasting to FloatTensor as it is compatible with CUDA
        datapoint['output_image']=datapoint['output_image'].type(torch.FloatTensor)
        if torch.cuda.is_available(): #move to gpu if available
                input_image = Variable(datapoint['input_image'].cuda()) #Converting a Torch Tensor to Autograd Variable
                output_image = Variable(datapoint['output_image'].cuda())
        else:
                input_image = Variable(datapoint['input_image'])
                output_image = Variable(datapoint['output_image'])

        optimizer.zero_grad()  #https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
        outputs = model(input_image)
        loss = criterion(outputs, output_image)
        loss.backward() #Backprop
        optimizer.step()    #Weight update
        writer.add_scalar('Training Loss',loss.data[0]/10, iter)
        iter=iter+1
        if iter % 10 == 0 or iter==1:
            # Calculate Accuracy         
            test_loss = 0
            total = 0
            # Iterate through test dataset
            for j,datapoint_1 in enumerate(test_loader): #for testing
                datapoint_1['input_image']=datapoint_1['input_image'].type(torch.FloatTensor)
                datapoint_1['output_image']=datapoint_1['output_image'].type(torch.FloatTensor)
           
                if torch.cuda.is_available():
                    input_image_1 = Variable(datapoint_1['input_image'].cuda())
                    output_image_1 = Variable(datapoint_1['output_image'].cuda())
                else:
                    input_image_1 = Variable(datapoint_1['input_image'])
                    output_image_1 = Variable(datapoint_1['output_image'])
                
                # Forward pass only to get logits/output
                outputs = model(input_image_1)
                test_loss += criterion(outputs, output_image_1).data[0]
                total+=datapoint_1['output_image'].size(0)
            test_loss=test_loss/total   #sum of test loss for all test cases/total cases
            writer.add_scalar('Test Loss',test_loss, iter) 
            # Print Loss
            time_since_beg=(time.time()-beg)/60
            print('Iteration: {}. Loss: {}. Test Loss: {}. Time(mins) {}'.format(iter, loss.data[0]/10, test_loss,time_since_beg))
        if iter % 500 ==0:
            torch.save(model,'checkpoints/model_iter_'+str(iter)+'.pt')
            print("model saved at iteration : "+str(iter))