import torch
import model as m
import load_dataset
import matplotlib.pyplot as plt
import numpy as np

#model = torch.load("trained_model.pth")

model = torch.load("trained_model_random.pth")
model.eval()

#loader, dataset = load_dataset.load_dataset("./data", "test")
loader, dataset = load_dataset.load_dataset("./data", "train")

background_color = (0, 0, 0)
mask_color = (255, 0, 0)


def visualize_pred(seg):    
    image = np.zeros((224,224,3), dtype=np.uint8)#create empty image
    for x in range(224):
        for y in range(224):
            res = seg[0][x][y].detach().numpy()
            print(res)
            if res < 0.5: # background
                image[x,y] = background_color #fill it with background
            else: # object detected
                image[x,y] = mask_color 
    plt.imshow(image)
    plt.show()


def forward_image(image, model):
    X = image.view(1, 3, 224, 224)
    res = model.forward(X)[0]
    return res
    
index = 0
image, label = dataset[index]

res = forward_image(image, model)
visualize_pred(res)
