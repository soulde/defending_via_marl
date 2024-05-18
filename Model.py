from torch.nn import Module, Sequential, Linear, ReLU, Tanh
import torch
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(Module):
    def __init__(self, feature_in, feature_out, action_scale, action_bias):
        super(Actor, self).__init__()
        self.backbone = Sequential(layer_init(Linear(feature_in, 128)),
                                   ReLU(),
                                   layer_init(Linear(128, 128)),
                                   ReLU(),
                                   )
        self.mean_head = Sequential(layer_init(Linear(128, feature_out)),
                                    Tanh())

        self.register_buffer(
            "action_scale", torch.tensor(action_scale, dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "action_bias", torch.tensor(action_bias, dtype=torch.float32, requires_grad=False)
        )

    def forward(self, x):
        feature = self.backbone(x)
        mean = self.mean_head(feature)
        return mean * self.action_scale + self.action_bias


class CriticTwin(Module):
    def __init__(self, state_in, action_in, q_in):
        super(CriticTwin, self).__init__()

        self.model = Sequential(layer_init(Linear((state_in + action_in) * q_in, 128)),
                                ReLU(),
                                layer_init(Linear(128, 128)),
                                ReLU(),
                                layer_init(Linear(128, 2 * q_in)))

    def forward(self, x, a):
        feature = torch.cat([x, a], dim=-1).view(x.shape[0], -1)
        return self.model(feature).reshape(x.shape[0], -1, 2)


class Critic(Module):
    def __init__(self, state_in, action_in, q_in):
        super(Critic, self).__init__()
        self.model = Sequential(layer_init(Linear((state_in + action_in) * q_in, 128)),
                                ReLU(),
                                layer_init(Linear(128, 128)),
                                ReLU(),
                                layer_init(Linear(128, q_in)))

    def forward(self, x, a):
        feature = torch.cat([x, a], dim=-1).view(x.shape[0], -1)
        return self.model(feature).reshape(x.shape[0], -1, 1)
