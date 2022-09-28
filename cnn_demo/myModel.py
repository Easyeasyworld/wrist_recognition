import torch
import torch.nn as nn

class myCNN(nn.Module):
    def __init__(self, batch, drop, in_channels):
        super(myCNN, self).__init__()
        self.batch = batch
        self.drop = drop
        self.in_channels = in_channels

        self.modelname = 'cnn'

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.LayerNorm([20, 7]),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # nn.LayerNorm([20, 7]),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm2d(64),
            # nn.LayerNorm([18, 5]),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.fc1 = nn.Linear(64*203*1, 17)
        self.drop1 = nn.Dropout(self.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feat = torch.flatten(x, start_dim=1, end_dim=3) #与x = x.view(self.batch, -1)一个效果
        # print(x.shape)
        x = x.view(self.batch, -1)
        x = self.fc1(x)
        x = self.drop1(x)
        return x, feat

def get_net(modelname, batch, drop, in_channels):
    if modelname == 'cnn':
        net = myCNN(batch, drop, in_channels)
    # c初始化参数
    # for param in net.parameters():
    #     # nn.init.normal_(param, mean=0, std=0.01)
    #     torch.nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('relu'))
    #     # p = param
    # for m in net.modules():
    #     if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
    #         torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))
    return net