import torch

from models.utils import get_efficientnet_params
from models.backbone import StemBlock

if __name__ == "__main__":

    model_params = get_efficientnet_params("efficientnet-b0")
    print(model_params)

    model = StemBlock(model_params)

    x = torch.rand(size=(4, 3, 224, 224))

    y = model(x)

    # summary(model.to(torch.device("cpu")), input_size=(3, 224, 224))
    print(x)
    print(y)
    from IPython import embed

    embed()
