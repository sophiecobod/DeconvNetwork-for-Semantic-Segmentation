import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class conv_deconv(nn.Module):
    def __init__(self):
        super(conv_deconv, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),#add batchnorm after each convolution,
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.convblock4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                
        self.convblock5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),                       
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        #kernel with the same size as image 7 for each convolution we get one value 4096 times in output with 0 padding 
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.ReLU(True)
        )

        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, out_channels=4096, kernel_size=1), #kernel size 1 to match each pixel 
            nn.ReLU(True)
        )

        self.fc6_deconv = nn.Sequential(
            nn.ConvTranspose2d(4096, out_channels=512, kernel_size=7),
            nn.ReLU(True)
        )
        
        self.maxunpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                
        self.deconvblock5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.maxunpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        self.deconvblock4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.maxunpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                
        self.deconvblock3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                
        self.deconvblock2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
                
        self.deconvblock1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        #input size of the image 224x224
        x = self.convblock1(x)
        size1 = x.size()
        x, ind1 = self.maxpool1(x)

        #input size of the image 112x112
        x = self.convblock2(x)
        size2 = x.size()
        x, ind2 = self.maxpool2(x)

        #input size of the image 56x56
        x = self.convblock3(x)
        size3 = x.size()
        x, ind3 = self.maxpool3(x)

        #input size of the image 28x28
        x = self.convblock4(x)
        size4 = x.size()
        x, ind4 = self.maxpool4(x)

        #input size of the image 14x14
        x = self.convblock5(x)
        size5 = x.size()
        x, ind5 = self.maxpool5(x)

        #input size of the image 7x7
        x = self.fc6(x)
        
        #input size of the image 1x1
        x = self.fc7(x)

        #input size of the image 1x1
        x = self.fc6_deconv(x)
        
        #input size of the image 7x7
        x = self.maxunpool5(x, ind5, size5)
        #input size of the image 14x14
        x = self.deconvblock5(x)

        x = self.maxunpool4(x, ind4, size4)
        #input size of the image 28x28
        x = self.deconvblock4(x)

        x = self.maxunpool3(x, ind3, size3)
        #input size of the image 56x56
        x = self.deconvblock3(x)

        x = self.maxunpool2(x, ind2, size2)
        #input size of the image 112x112
        x = self.deconvblock2(x)

        x = self.maxunpool1(x, ind1, size1)
        #input size of the image 224x224
        x = self.deconvblock1(x)
    
        return x

# to do some experiment
# to do some experiment
#import the weights of VGG16 to do transfer learning (source : dataflowr notebook)
if __name__ == "__main__":

    model = conv_deconv()
    w = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')

    for (name, child) in model.convblock1.named_children():
        if name == '0':
            print(child)
            child.weight.data = w['features.0.weight']
            child.bias.data = w['features.0.weight']


"""
Pour la suite:
- regarder comment sont mis en place les poids dans vgg
- les mettres sur chaque couches
- tester que les poids ont bien changée etc

- Initiliser gaussien les autres poids

"""
    """
    i = 0
    for block in model.children():
        for idy, msub in enumerate(m.children()):
            classname = msub.__class__.__name__
            print(classname)
            continue
            if classname.find('Conv') != -1:
                msub.weight.data = w['features.'+str(i)+'.weight']#.clone()
                msub.bias.data = w['features.'+str(i)+'.bias']#.clone()
                print(msub,'features.'+str(i))
            i +=1

"""

"""

layers = ['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 'features.5.weight', 'features.5.bias', 'features.7.weight', 'features.7.bias', 'features.10.weight', 'features.10.bias', 'features.12.weight', 'features.12.bias', 'features.14.weight', 'features.14.bias', 'features.17.weight', 'features.17.bias', 'features.19.weight', 'features.19.bias', 'features.21.weight', 'features.21.bias', 'features.24.weight', 'features.24.bias', 'features.26.weight', 'features.26.bias', 'features.28.weight', 'features.28.bias', 'classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias', 'classifier.6.weight', 'classifier.6.bias']
"""