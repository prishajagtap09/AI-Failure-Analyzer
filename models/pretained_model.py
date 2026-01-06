import torch
import torchvision.transforms as transforms
from torchvision import models

class PretrainedModel:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):
        """
        image: PIL Image
        returns: probabilities tensor
        """
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        return probabilities.cpu()
