import torch
from torch import nn
from torchvision.models import get_model, get_model_weights, get_weight, list_models


available_models = list_models()

class SimpleResnet(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        weights: str = "DEFAULT",
        output_shape: list = [68, 2],
    ):
        super().__init__()
        model = get_model(model_name, weights=weights)
        supported = False
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, output_shape[0] * output_shape[1])
            supported = True
        elif hasattr(model, 'heads'):
            heads = getattr(model, 'heads')
            if hasattr(heads, 'head'):
                num_ftrs = model.heads.head.in_features
                model.heads.head = nn.Linear(num_ftrs, output_shape[0] * output_shape[1])
                supported = True

        if not supported:
            print("Model is not supported")
            exit(1)
        self.model = model
        self.output_shape = output_shape

    def forward(self, x):
        # batch_size, channels, width, height = x.size()

        return self.model(x).reshape(x.size(0), self.output_shape[0], self.output_shape[1])


if __name__ == "__main__":
    print(available_models)
    m = SimpleResnet()
    output = m(torch.randn(1, 3, 224, 224))
    print(output.shape)