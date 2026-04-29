from torch.nn import Module, ModuleList, Linear, ReLU, LeakyReLU, Sequential, LayerNorm
from src.CauchyActivation import CauchyActivation


class MLPConnector(Module):
    def __init__(self, batch_size, hidden_one=512, hidden_two=2048, hidden_three=4096):
        super(MLPConnector, self).__init__()

        self.linear1 = Linear(hidden_one, hidden_two)
        self.linear2 = Linear(hidden_two, hidden_two)
        self.linear3 = Linear(hidden_two, hidden_three)
        self.linear4 = Linear(hidden_three, hidden_three)


        self.batch_size = batch_size
        self.hidden_layers = [hidden_one, hidden_two, hidden_three]

        self.output_shape = int((self.batch_size*self.hidden_layers[-1]/256)), 256
        self.layer_norms = ModuleList([LayerNorm(hidden_one), LayerNorm(hidden_two), LayerNorm(hidden_three)])

    def forward(self, x):

        # Intermediate MLP layers
        x = self.layer_norms[0](x)
        x = self.linear1(x)
        x = LeakyReLU()(x)

        x = self.layer_norms[1](x)
        x = self.linear2(x)
        x = LeakyReLU()(x)

        # Output head MLP
        x = self.layer_norms[1](x)
        x = self.linear3(x)
        x = LeakyReLU()(x)

        x = self.layer_norms[2](x)
        x = self.linear4(x)
        output = x.view(self.output_shape)

        return output