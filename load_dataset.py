import torch 
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms 
import voc

def load_dataset(datapath, dataset_choice):

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
            transforms.ToTensor()])
    transform_label = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor()])
    dataset = voc.VOCSegmentation(root=datapath, year='2012', image_set=dataset_choice,
                                                    download=True, transform = transform, target_transform=transform_label)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                            shuffle=True, num_workers=2)


    return loader, dataset

if __name__ == "__main__":
    train_loader, train_dataset = load_dataset("./data", "train")
    print(train_loader)
    print(train_dataset)
    print(len(train_dataset))
    print(train_dataset[5])
    print(type(train_dataset[5]))

    print("nb batches:")
    print(len(list(train_loader)))
