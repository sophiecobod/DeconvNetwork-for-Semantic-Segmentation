import torch
import model
import load_dataset
import matplotib.pyplot as plt
import numpy as np

model = torch.load("trained_model.pth")
model.eval()

#loader, dataset = load_dataset.load_dataset("./data", "test")
loader, dataset = load_dataset.load_dataset("./data", "train")

background_color = (0, 0, 0)
mask_color = (255, 0, 0)


def visualize_pred(seg):    
    image = np.zeros((224,224,3), dtype=np.int8)
    for x in range(224):
        for y in range(224):
            image[x,y] = seg[0][x][y]
    plt.imshow(image)
    plt.show()


def forward_image(image, model):
    X = image.view(1, 3, 224, 224)
    res = conv_model.forward(X)[0]
    return res
    
index = 0
image, label = dataset[index]
conv_model = model.create_model()

res = test_image(image, conv_model)
visualize_pred(res)
