import torch
import torchvision
import torchvision.transforms as transforms

def get_model(name, num_classes=10, pretrained=False, batch_size=64, image_size=32, **kwargs):
    if name == 'cifar10':
        # The transform should match the image_size parameter
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Download and load the training data
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        
        # Create the DataLoader, which is what the script actually needs
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)
        
        return train_loader
    else:
        raise ValueError(f"Model {name} not supported by this simplified ModelLoader.")
