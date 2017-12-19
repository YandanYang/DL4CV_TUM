"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        model_pre=torchvision.models.vgg16_bn(pretrained=True)
        model_pre=model_pre.features
        self.relu = nn.ReLU(inplace=True)
        
        self.pretrained=model_pre
        self.deconv1=nn.ConvTranspose2d(512,512,kernel_size=3, stride=2,padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2=nn.ConvTranspose2d(512, 256,kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3=nn.ConvTranspose2d(256, 128,kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4=nn.ConvTranspose2d(128, 64 ,kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5=nn.ConvTranspose2d(64, 64 ,kernel_size=3, stride=2,padding=1,output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv1=nn.Conv2d(64,num_classes,kernel_size=1)
        ##self.conv1=nn.Conv2d(512,num_classes,kernel_size=1)
       # self.bn = nn.BatchNorm2d(num_classes)
        ##self.deconv1=torch.nn.Upsample((240,240), mode='nearest')

        #self.deconv1=torch.nn.Upsample(scale_factor=35, mode='nearest')
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        ##x=self.pretrained(x)
        ##x=self.conv1(x)
        ##x=self.deconv1(x)
        ##x=x[:,:,2:-3,2:-3]
    
       #x=self.bn(x)
    #low layer infomation
        for i in range(0,24):
            x=self.pretrained[i](x)
        x1=x
        for i in range(24,34):
            m=self.pretrained[i]
            x=m(x)
        x2=x
        for i in range(34,44):
            x=self.pretrained[i](x)
         
        #print(x.size())
        #upsampling
        x=self.deconv1(x) 
        #print(x.size())
        x=self.bn1(self.relu(x)+x2)
        #print(x.size())
        x=self.deconv2(x)
        x=self.bn2(self.relu(x)+x1)
        x=self.deconv3(x)
        x=self.bn3(self.relu(x))
        x=self.deconv4(x)
        x=self.bn4(self.relu(x))
        x=self.deconv5(x)
        x=self.bn5(self.relu(x))
        #
        x=self.conv1(x)
        torch.nn.Upsample((240,240), mode='nearest')
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
