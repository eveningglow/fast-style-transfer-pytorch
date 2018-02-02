import torch
import torchvision.datasets as Datasets
import torchvision.transforms as Transforms

def data_loader(root, batch_size=4):
    transform = Transforms.Compose([Transforms.Scale(256), 
                                    Transforms.CenterCrop(256),
                                    Transforms.ToTensor()
                                   ])
    dataset = Datasets.ImageFolder(root, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    length = len(dataset)
    
    return loader, length