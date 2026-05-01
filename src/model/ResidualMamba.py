from torch.nn import Module, LayerNorm, RMSNorm, GELU, Dropout
from mamba_ssm import Mamba

class ResidualMamba(Module):
    def __init__(self, d_model=512, d_state=16, d_conv=4, expand=2):
        super(ResidualMamba, self).__init__()

        #self.layer_norm = RMSNorm(d_model)
        self.layer_norm = LayerNorm(d_model)
        self.mamba = Mamba(d_model, d_state, d_conv, expand)
        self.dropout = Dropout(0.1)

    def forward(self, x, mask):

        residual = x
        x = self.layer_norm(x)
        x = self.mamba(x)
        x = GELU()(x)
        x = self.dropout(x)
        x = x * mask

        return x + residual, mask

