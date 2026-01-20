from torch import nn

# class cnn(nn.Module):
#     # def __init__(self, input_channel = 1, hidden_channels = 64, kernel_size = 64, stride = 8, output_channels = 16, feature_length = 4):
#     def __init__(self, input_channel = 1, hidden_channels = 64, kernel_size = 5, stride = 1, output_channels = 128, feature_length = 5):
#         super(cnn, self).__init__()

#         self.conv_block1 = nn.Sequential(
#             nn.Conv1d(input_channel, hidden_channels, kernel_size=kernel_size,
#                       stride=stride, bias=False, 
#                     # padding=(27)
#                       padding=(kernel_size // 2)
#                       ),
#             nn.BatchNorm1d(hidden_channels),
#             nn.ReLU(),
#             # nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1),

#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=8, stride=1, bias=False,
#                       padding=3),
#             nn.BatchNorm1d(hidden_channels // 2),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
#             # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv1d(hidden_channels // 2, output_channels, kernel_size=8, stride=1, bias=False,
#                       padding=3),
#             nn.BatchNorm1d(output_channels),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
#             # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         )

#         self.adaptive_pool = nn.AdaptiveAvgPool1d(feature_length)

#     def forward(self, x_in):
#         x = self.conv_block1(x_in)
#         # print(x.shape)
#         x = self.conv_block2(x)
#         # print(x.shape)
#         x = self.conv_block3(x)
#         # print(x.shape)
#         x = self.adaptive_pool(x)
#         # print(x.shape)

#         x_flat = x.reshape(x.shape[0], -1)  #1*640
#         # print(x_flat.shape)

#         return x_flat
    
class cnn(nn.Module):
    def __init__(self, input_channel= 1, hidden_channels = 64, kernel_size = 5, stride = 1, output_channels = 128, feature_length = 5):
        super(cnn, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channel, hidden_channels, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
   
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, output_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(feature_length)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat
# model = cnn()
# import torch
# x = torch.rand((1,1,1024))
# model(x)
# model = cnn()
# for m in model.modules():
#     print(m)