import model
import load_dataset
import torch 

loader, dataset = load_dataset.load_dataset("./data", "train")
x,y = dataset[0]
X = x.view(1, 3, 224, 224)

conv_model = model.create_model()
res = conv_model.forward(X)[0]

#print(conv_model)

#print(res)
#print(res.shape)

#function to compute accuracy 
def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    binary_preds = (preds >= 0.5).long()
    valid = (label >= 0.5).long()
    acc_sum = torch.sum(valid * (binary_preds == label.long()).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

print(res.shape)
print(y.shape)

acc = pixel_acc(res, y)
print(acc)