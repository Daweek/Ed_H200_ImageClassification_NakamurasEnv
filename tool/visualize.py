import timm
from timm.models import create_model
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import torchvision
import torchvision.transforms as transforms
from timm.data import create_transform
import argparse


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape
    print(n,c,w,h)

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default="/groups/gaa50131/user/nakamura/Fractal_script/ImageClassification/outputs/rndb_gene/resnet50/random_noise/1000/3/pretrain/0621_205358"
)

parser.add_argument(
    "--model",
    "-m",
    type=str,
    default="resnet50",
    choices=["resnet50", "vit_tiny_patch16_224"],
    help="Choose architecture.",
)

args = parser.parse_args()

## model

model = create_model("resnet50", pretrained=False)
print(model.conv1.weight.data.clone())
model.load_state_dict(torch.load(f"{args.path}/resnet50_random_noise_900ep.pth", map_location='cpu')['model'])


#filterのかしか
layer = 1
filter =  model.conv1.weight.data.clone()
visTensor(filter, ch=0, allkernels=False)


plt.axis('off')
plt.ioff()
plt.show()
plt.savefig(f"{args.path}/load_param_filter.png")
plt.close()


#hitst
plt.hist(model.conv1.weight.data.clone().reshape(-1).numpy(),bins=100)
plt.show()
plt.savefig(f"{args.path}/load_param_hist.png")
plt.close()


# model = create_model("resnet50", pretrained=False)
# # model.load_state_dict(torch.load("resnet50_random_noise_900ep.pth", map_location='cpu')['model'])
# layer = 1
# filter =  model.conv1.weight.data.clone()
# visTensor(filter, ch=0, allkernels=False)

# plt.axis('off')
# plt.ioff()
# plt.show()



##
model = model.cuda()
transform = create_transform(
    224,
    is_training=False,
    use_prefetcher=False,
    no_aug=True,
    scale=(1.0,1.0),
    ratio=(1.0,1.0),
    hflip=0.0,
    vflip=0.,
    color_jitter=0.0,
    interpolation='bicubic',
    mean=( 0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    re_prob=0,
    re_mode='pixel',
    re_count=1
    )
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=4)

output=[]
label_ls=[]
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images , labels = images.cuda(), labels.cuda()
        print(i,labels.shape)
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.act1(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.global_pool(x)
        
        x= x.cpu()
        labels= labels.cpu()
        output.append(x.clone().numpy())  
        label_ls.append(labels.clone().numpy())
        if (i+1) ==5000:
            break

output = np.array(output)
label = np.array(label_ls).reshape(len(output),-1)

#t-SNEで次元削減
import pandas as pd
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
X_embedded = tsne.fit_transform(output.reshape(len(output),-1))

colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink"]
cate = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize = (10, 10))
for i in range(len(X_embedded)):
    plt.scatter(X_embedded[i,0],  
                X_embedded[i,1],
                label=cate[label[i,0]-1],
                color = colors[label[i,0]-1]
               )

plt.legend( cate )
plt.savefig(f"{args.path}/load_param_t-sne.png")
plt.close()
print("ploted t-sne's visualization")

#hitst
plt.hist(output.reshape(-1),bins=500)
plt.show()
plt.savefig(f"{args.path}/load_param_output_hist.png")
plt.close()
print("ploted output's histgram")


##################### random param ##################

model = create_model("resnet50", pretrained=False)
print(model.conv1.weight.data.clone())


#filterのかしか
layer = 1
filter =  model.conv1.weight.data.clone()
visTensor(filter, ch=0, allkernels=False)


plt.axis('off')
plt.ioff()
plt.show()
plt.savefig(f"{args.path}/random_param_filter.png")
plt.close()


#hitst
plt.hist(model.conv1.weight.data.clone().reshape(-1).numpy(),bins=100)
plt.show()
plt.savefig(f"{args.path}/random_param_hist.png")
plt.close()


model = model.cuda()

output=[]
label_ls=[]
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(testloader):
        images , labels = images.cuda(), labels.cuda()
        print(i,labels.shape)
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.act1(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.global_pool(x)
        
        x= x.cpu()
        labels= labels.cpu()
        output.append(x.clone().numpy())  
        label_ls.append(labels.clone().numpy())
        if (i+1) ==5000:
            break

output = np.array(output)
label = np.array(label_ls).reshape(len(output),-1)

#t-SNEで次元削減
tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
X_embedded = tsne.fit_transform(output.reshape(len(output),-1))

colors =  ["r", "g", "b", "c", "m", "y", "k", "orange","pink"]
cate = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize = (10, 10))
for i in range(len(X_embedded)):
    plt.scatter(X_embedded[i,0],  
                X_embedded[i,1],
                label=cate[label[i,0]-1],
                color = colors[label[i,0]-1]
               )

plt.legend( cate )
plt.savefig(f"{args.path}/random_param_t-sne.png")
plt.close()
print("ploted t-sne's visualization")

#hitst
plt.hist(output.reshape(-1),bins=500)
plt.show()
plt.savefig(f"{args.path}/random_param_output_hist.png")
plt.close()
print("ploted output's histgram")