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

def create_model():
    model = conv_deconv()
    w = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')

    def init_vgg_weights_block(block, name_list, verbose=False):
        i = 0
        for (name, child) in block.named_children():
                classname = child.__class__.__name__
                if classname.find('Conv') != -1:
                    feat_name = 'features.'+name_list[i]+'.weight'
                    bias_name = 'features.'+name_list[i]+'.bias'

                    child.weight.data = w[feat_name]
                    child.bias.data = w[bias_name]
                    if verbose:
                        print(feat_name + " and " + bias_name + " initialized.")

                    i += 1

    def init_gaussian_weights(model, verbose=False):
        for block in model.children():
            for msub in block.children():
                classname = msub.__class__.__name__
                if classname.find('ConvTranspose') != -1:
                    msub.weight.data.normal_(0, 0.01)
                    msub.bias.data.fill_(0)
                    if verbose:
                        print(classname + " initialized.")


    init_vgg_weights_block(model.convblock1,["0", "2"])
    init_vgg_weights_block(model.convblock2,["5", "7"])
    init_vgg_weights_block(model.convblock3,["10", "12", "14"])
    init_vgg_weights_block(model.convblock4,["17", "19", "21"])
    init_vgg_weights_block(model.convblock5,["24", "26", "28"])

    init_gaussian_weights(model)

    return model

# to do some experiment
# to do some experiment
#import the weights of VGG16 to do transfer learning (source : dataflowr notebook)
if __name__ == "__main__":
    pass
