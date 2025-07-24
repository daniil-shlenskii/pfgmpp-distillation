import torch.nn as nn


def preprocess_edm_net(
    net: nn.Module,
    remove_dropout: bool = True,
):
    if hasattr(net.model, 'map_augment'):
        net.model.map_augment = None

    if remove_dropout:
        remove_dropout_from_model(net)

    return net

class EvalModeDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return x / (1 - self.p)

    def extra_repr(self):
        return f"p={self.p}"

def remove_dropout_from_model(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, nn.Dropout):
            setattr(model, name, EvalModeDropout(p=module.p))
        elif isinstance(module, nn.Module):
            remove_dropout_from_model(module)
