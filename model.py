import torch.nn as nn


class VGG(nn.Module):
    def __init__(self,out_dim):
        super(VGG, self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 224, 224]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, 1, 1),  # [64, 224, 224]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 112, 112]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 112, 112]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, 1, 1),  # [128, 112, 112]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 56, 56]


            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),  # [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),  # [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 28, 28]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [512, 14, 14]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [512,7, 7]
        )

        self.fc=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(4096,out_dim),
            nn.Dropout(0.5),
            nn.ReLU(),

        )

    def forward(self,x):
        out=self.cnn(x)
        out=out.view(out.size()[0],-1)
        return self.fc(out)


class CNN(nn.Module):
    def __init__(self,backbone_out_dim,out_dim):
        super(CNN, self).__init__()
        self.backbone=VGG(backbone_out_dim)
        self.fc=nn.Sequential(
            nn.Linear(backbone_out_dim,out_dim)
        )


    def forward(self,x):
        out=self.backbone(x)
        return self.fc(out)
