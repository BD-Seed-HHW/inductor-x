import torch
import torch_npu
import inductor_npu
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(-1)


if __name__ == "__main__":
    net = Net()
    compiled_net = torch.compile(net, backend="inductor")

    input = torch.randn((1, 1, 7168)).npu()

    output = net(input)
    output1 = compiled_net(input)
    torch.testing.assert_allclose(output, output1, rtol=1e-03, atol=1e-03)