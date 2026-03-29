import torch
import torch.nn as nn

class ARConv(nn.Module):

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9):
        super(ARConv, self).__init__()
        self.l_max = l_max
        self.w_max = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride


        self.size_list = [(3,3), (3,5), (5,3), (3,7), (7,3), (5,5), (5,7), (7,5), (7,7)]

        self.convs = nn.ModuleList([
            nn.Conv2d(inc, outc, kernel_size=sz, stride=1, padding=(sz[0]//2, sz[1]//2))
            for sz in self.size_list
        ])

        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )

        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )

        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.size()

        l_map = self.l_conv(x)  # (B,1,H,W)
        w_map = self.w_conv(x)  # (B,1,H,W)

        mean_l = l_map.mean()
        mean_w = w_map.mean()

        N_X = int(mean_l.item() * (self.l_max - 1)) + 1
        N_Y = int(mean_w.item() * (self.w_max - 1)) + 1

        if N_X % 2 == 0:
            N_X = N_X - 1 if N_X > 3 else 3
        if N_Y % 2 == 0:
            N_Y = N_Y - 1 if N_Y > 3 else 3
        N_X = min(max(N_X, 3), 7)
        N_Y = min(max(N_Y, 3), 7)

        try:
            idx = self.size_list.index((N_X, N_Y))
        except ValueError:
            idx = 0  
        conv_output = self.convs[idx](x)

        m = self.m_conv(x)
        b = self.b_conv(x)

        out = conv_output * m + b
        return out
