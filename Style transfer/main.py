from functools import total_ordering
from matplotlib import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# feature extractor
class VGG(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGG, self).__init__()

        self.chosen_layers = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        """Extracting feature maps"""
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)

            if name in self.chosen_layers:
                features.append(x)
        return features
        

def load_image(image_path):
    """Load an image and convert it to a Tensor"""
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
im_size = 300
loader = transforms.Compose(
    [
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
)

content = load_image("content.jpg")
style = load_image("style.jpg")

generated = content.clone().requires_grad_(True)
model = VGG().to(device).eval()

TOTAL_STEPS = 1800
LR = 0.001
ALPHA = 1
BETA = 0.025
optimizer = optim.Adam([generated], lr=LR)

for step in tqdm(range(TOTAL_STEPS+1)):
    # extracting multiple conv feature vectors
    generated_features = model(generated)
    content_features = model(content)
    style_features = model(style)

    style_loss = content_loss = 0

    for gen_f, con_f, style_f in zip(generated_features, style_features, style_features):
        # compute content loss between generated image and content image
        content_loss += torch.mean((gen_f - con_f) ** 2) 
        
        batch_size, channel, height, width = gen_f.shape
        # Gram Matrix
        G = gen_f.view(channel, height*width).mm(
            gen_f.view(channel, height*width).t()
        )

        S = style_f.view(channel, height*width).mm(
            style_f.view(channel, height*width).t()
        )
        # compute style loss between generated image and style image
        style_loss += torch.mean((G - S) ** 2 / (channel * height * width))
    
    total__loss = ALPHA * content_loss + BETA * style_loss
    optimizer.zero_grad()
    total__loss.backward()
    optimizer.step()

    if (step + 1) % 200 == 0:
        print(f"step = {step}, loss = {total__loss}")
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = generated.clone().squeeze()
        # img = denorm(img).clamp_(0, 1)
        save_image(img, f"generated/{str(step+1)}_generated.png")
        