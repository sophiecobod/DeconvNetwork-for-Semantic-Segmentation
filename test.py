import model
import load_dataset

loader, dataset = load_dataset.load_dataset("./data", "train")
x,y = dataset[0]
X = x.view(1, 3, 224, 224)

conv_model = model.conv_deconv()
res = conv_model.forward(X)

print(conv_model)

print(res)
print(res.shape)