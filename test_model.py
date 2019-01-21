import torch
import model as m
import load_dataset
import matplotlib.pyplot as plt
import numpy as np

#model = torch.load("trained_model.pth")

model = torch.load("trained_model.pth")
model.eval()

#loader, dataset = load_dataset.load_dataset("./data", "test")
loader, dataset = load_dataset.load_dataset("./data", "train")

background_color = (0, 0, 0)
mask_color = (255, 255, 255)


def visualize_pred(seg):    
    image = np.zeros((224,224,3), dtype=np.uint8)#create empty image
    image_005 = np.zeros((224,224,3), dtype=np.uint8)#create empty image
    
    for x in range(224):
        for y in range(224):
            res = seg[0][x][y].cpu().detach().numpy()
            print(res)
            if res < 0.05:
                image_005[x,y] = background_color #fill it with background
            else:
                image_005[x,y] = mask_color
                
            if res < 0.5: # background
                image[x,y] = background_color #fill it with background
            else: # object detected
                image[x,y] = mask_color
                
    plt.imsave("visu_normal.png", image)
    plt.imsave("visu_005.png", image_005)
    #plt.show()


device = torch.device('cuda')

def forward_image(image, model):
    X = image.view(1, 3, 224, 224)
    X = X.to(device)
    res = model.forward(X)[0]
    return res
    
index = 0
image, label = dataset[index]

res = forward_image(image, model)
visualize_pred(res)
