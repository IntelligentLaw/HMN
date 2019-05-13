from model.BasicModule import *
from model.model_utils import LayerNorm

class FCLayer(BasicModule):
    def __init__(self, input_dim, output_dim, linear_hidden_dim = 1024, type = "normal"):
        super(FCLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_hidden_dim = linear_hidden_dim
        if type == "normal":
            self.fc = nn.Linear(self.input_dim, self.output_dim)
        elif type == "deep":
            self.fc = nn.Sequential(
                nn.Linear(self.input_dim, self.linear_hidden_dim),
                nn.BatchNorm1d(self.linear_hidden_dim),
                # LayerNorm(self.linear_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.linear_hidden_dim, self.output_dim)
            )


    def forward(self, inputs):
        logits = self.fc(inputs)
        return logits