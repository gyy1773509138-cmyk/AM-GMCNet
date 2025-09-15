from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset_path, img_size=150, batch_size=32, test_ratio=0.1):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomChoice([
            transforms.RandomRotation(0),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.RandomRotation(270)
        ]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )
