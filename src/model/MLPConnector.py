from torch.nn import Module, ModuleList, Linear, ReLU, LeakyReLU, LayerNorm, RMSNorm, Dropout

class MLPConnector(Module):
    def __init__(self, batch_size, hidden_one=512, hidden_two=4096):
        super(MLPConnector, self).__init__()

        self.linear1 = Linear(hidden_one, hidden_one)
        self.linear2 = Linear(hidden_one, hidden_two)
        self.linear3 = Linear(hidden_two, hidden_two)


        self.batch_size = batch_size
        self.hidden_layers = [hidden_one, hidden_two]

        self.output_shape = int((self.batch_size*self.hidden_layers[-1]/256)), 256
        self.layer_norms = ModuleList([LayerNorm(x) for x in self.hidden_layers])
        self.dropout = Dropout(0.55)

    def forward(self, x):

        # Intermediate MLP layers
        x = self.layer_norms[0](x)
        x = self.linear1(x)
        x = LeakyReLU()(x)
        x = self.dropout(x)

        x = self.layer_norms[0](x)
        x = self.linear2(x)
        x = LeakyReLU()(x)
        x = self.dropout(x)

        # Output head MLP

        x = self.layer_norms[1](x)
        x = self.linear3(x)
        output = x.view(self.output_shape)

        return output