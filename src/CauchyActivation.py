import torch


class CauchyActivation(torch.nn.Module):
    def __init__(self, initial_lambda=1.0, initial_lambda2=1.0, initial_d=1.0):
        super(CauchyActivation, self).__init__()

        self.lambda1 = torch.nn.Parameter(torch.tensor(initial_lambda, dtype=torch.float32))
        self.lambda2 = torch.nn.Parameter(torch.tensor(initial_lambda2, dtype=torch.float32))
        self.d = torch.nn.Parameter(torch.tensor(initial_d, dtype=torch.float32))

    def forward(self, x):
        x2_d2 = x**2 + self.d**2
        return ((self.lambda1 * x)/x2_d2) + (self.lambda2/x2_d2)