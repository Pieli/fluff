import torch
from cnn_cifar10 import CNN
from torch import nn, optim


class FedAD:
    """
    Aggregator: FedAD
    Authors: Gong et al.
    Year: 2021
    """

    def __init__(self, config=None, **kwargs):
        pass

    def run_aggregation(self, models: list[nn.Module]):
        model = models[0]

        model.register_backward_hook(backward_hook, prepend=False)
        model.register_forward_hook(forward_hook, prepend=False)

        # for each distillation step t = 1, ... T do

        # random subset (gamma-fraction) from K locals
        # K_t =

        # a batch of public data from D_0 with size S
        # x_0 =

        # for each  k \in K do
        # z_k, A_k  = f(x_0; theta_k)

        # ensemble logits

        # for each class
        # sample a subset of K_t

        # I_c, U_c ensemble

        # server model (z, A_Z)

        # update server model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Backward pass for the target class
        target = output[:, class_idx].sum()
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam / cam.max()  # Normalize
        return cam


# TODO: Class?!
class kl_loss(torch.nn.Module):
    def __init__(self, T=3, singlelabel=False):
        super().__init__()
        self.T = T
        self.singlelabel = singlelabel
        self.criterion = torch.nn.KLDivLoss(reduction="batchmean")


# Compute Grad-CAM for a local model


def compute_attention_map(model, data, class_idx, target_layer):
    grad_cam = GradCAM(model, target_layer)
    attention_map = grad_cam.generate(data, class_idx)
    return attention_map


# Compute Intersection and Union of Attention Maps


def compute_intersection_union(attention_maps):
    attention_intersection = torch.min(torch.stack(attention_maps), dim=0)[0]
    attention_union = torch.max(torch.stack(attention_maps), dim=0)[0]
    return attention_intersection, attention_union


# Attention-based Loss


def attention_loss(attention_student, intersection, union):
    # Enforce consensus with intersection
    loss_intersection = torch.mean(intersection * attention_student)
    # Encourage diversity with union
    loss_union = torch.mean(union * attention_student)
    return -loss_intersection + loss_union


# Distill with Grad-CAM attention


def distill_with_attention(central_model, local_models, public_data, target_layer):
    central_optimizer = optim.SGD(central_model.parameters(), lr=0.01)
    for data in public_data:
        central_logits = central_model(data)
        central_attention = compute_attention_map(central_model, data, class_idx=0, target_layer=target_layer)

        local_attentions = [
            compute_attention_map(local, data, class_idx=0, target_layer=target_layer) for local in local_models
        ]

        intersection, union = compute_intersection_union(local_attentions)

        # Compute Attention Loss
        loss_att = attention_loss(central_attention, intersection, union)

        # Backpropagation
        central_optimizer.zero_grad()
        loss_att.backward()
        central_optimizer.step()


if __name__ == "__main__":
    m_1 = CNN()
    m_1.load_state_dict(torch.load("models/model_1.pth", map_location=torch.device("cpu")))

    models = [m_1]
    fedad = FedAD(config=None)
    fedad.run_aggregation(models)
