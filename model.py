import torch
import torch.nn as nn

# ---------- Residual Dense Network blocks ----------
class RDB_Conv(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, growth, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDNBlock(nn.Module):
    def __init__(self, in_ch, num_layers=4, growth=32):
        super().__init__()
        ch = in_ch
        self.layers = nn.Sequential(*[RDB_Conv(ch+i*growth, growth) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_ch + num_layers*growth, in_ch, 1)

    def forward(self, x):
        out = self.layers(x)
        return self.lff(out) + x

class RDN(nn.Module):
    def __init__(self, in_ch, blocks=3, layers=4, growth=32):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.body = nn.Sequential(*[RDNBlock(64, layers, growth) for _ in range(blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        return self.tail(x)

# ---------- encoder/decoder wrappers ----------
class RDNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RDN(in_ch=6)

    def forward(self, cover, watermark):
        return self.net(torch.cat([cover, watermark], 1))

class RDNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RDN(in_ch=3)

    def forward(self, x):
        return self.net(x)
