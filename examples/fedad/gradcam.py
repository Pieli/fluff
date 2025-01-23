import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from cnn_cifar10 import CNN
from torch.utils.data import DataLoader


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
                print(name, module)
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        pred = output.argmax(dim=1).item()

        if pred != class_idx:
            print(f"[IMPORTANT] The model predicted {pred} but the target class is {class_idx}")

        # Select the target class
        target = output[:, class_idx].sum()

        # Backward pass
        target.backward(retain_graph=True)

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(0, 2, 3), keepdim=True)

        # weighted average
        cam = (weights * self.activations).sum(dim=1)

        # only account for positive change
        cam = torch.relu(cam)

        # normalize the values
        cam = cam / cam.max()  # Normalize

        return cam


def visualize_gradcam(input_tensor, gradcam):
    if type(input_tensor) is torch.Tensor:
        # Extract the first image tensor
        # Shape: (C, H, W)
        image_tensor = input_tensor[0]

        # Convert to NumPy array and rearrange dimensions to H x W x C
        image_numpy = image_tensor.permute(1, 2, 0).numpy()
    else:
        # numpy case
        image_numpy = input_tensor

    # Convert to BGR format (OpenCV uses BGR, while the dataset uses RGB)
    image_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)

    # Normalize the pixel values to the range [0, 255] if needed
    img = (image_bgr * 255).astype(np.uint8)

    # Resize the heatmap to match the image dimensions
    heatmap = cv2.resize(gradcam.detach().numpy(), (img.shape[1], img.shape[0]))

    # Convert the heatmap to the range [0, 255]
    heatmap = np.uint8(255 * heatmap)

    # Apply the colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Merge heatmap and the original image
    # superimposed_img = heatmap * 0.5 + img
    superimposed_img = cv2.addWeighted(img, 0.8, heatmap, 0.8, 0)

    # Save the result
    cv2.imwrite("./map.jpg", superimposed_img)
    cv2.imwrite("./heatmap.jpg", heatmap)


class IndexedCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        # Get the image and label from the original dataset
        img, label = super().__getitem__(index)

        # Return the index as well
        return img, label, index


def get_loader():
    # Define transformations (if needed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the training set
    trainset = IndexedCIFAR10(root="./data", train=True, download=True, transform=transform)

    # Create a Loader
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True)

    return train_loader


if __name__ == "__main__":
    loader = get_loader()

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

    m_1 = CNN()
    try:
        m_1.load_state_dict(
            torch.load(
                "experiements/models/model_1.pth",
                map_location=torch.device("cpu"),
            )
        )
    except FileNotFoundError:
        print("[INFO] file not found: using the local directory instead.")
        m_1.load_state_dict(
            torch.load(
                "models/model_1.pth",
                map_location=torch.device("cpu"),
            )
        )

    m_1.eval()

    cam = GradCAM(m_1, "conv3")

    input_tensor, target_label, ind = next(iter(loader))

    # Generate Grad-CAM
    gradcam_map = cam.generate(input_tensor, int(target_label))

    # plt.matshow(gradcam_map.detach().numpy().squeeze())
    # plt.show()

    # Get the original image (without transforms)
    orig_image, _ = train_set.__getitem__(ind)

    orig_num = np.array(orig_image.copy())
    colored = cv2.cvtColor(orig_num, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./orig.jpg", colored)

    # train_set.

    # Visualize
    orig_v2 = np.array(orig_image.copy())
    visualize_gradcam(orig_v2, gradcam_map.squeeze())
