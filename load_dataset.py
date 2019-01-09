import torch 
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms 
import voc

def load_dataset(datapath, dataset_choice):

    transform = transforms.Compose(
        [transforms.Resize((200, 200)),
            transforms.ToTensor()])
    transform_label = transforms.Compose(
        [transforms.Resize((200, 200)),
        transforms.ToTensor()])
    dataset = voc.VOCSegmentation(root=datapath, year='2012', image_set=dataset_choice,
                                                    download=True, transform = transform, target_transform=transform_label)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                            shuffle=True, num_workers=2)


    return loader

if __name__ == "__main__":
    train_loader = load_dataset("./data", "train")
    print(train_loader)