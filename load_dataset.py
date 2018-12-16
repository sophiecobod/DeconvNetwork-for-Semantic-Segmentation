import torch 
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms 

def load_dataset(datapath, dataset_choice):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.VOCSegmentation(root=datapath, year='2012', image_set=dataset_choice,
                                                    download=True, transform = transform)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                            shuffle=True, num_workers=2)


    return loader

if __name__ == "__main__":
    train_loader = load_dataset("./data", "train")
    print(train_loader)