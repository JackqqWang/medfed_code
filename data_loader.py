from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
import torch
import clip
import os
batch_size=16
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device)
# å‡è®¾ extra_matrix æ˜¯æ‚¨çš„ 10000x100 ç»´çŸ©é˜µ
logis_path="output_logits.pt"
# torch.save(output_logits, logis_path)
extra_matrix=torch.load(logis_path)

class CustomCIFAR100(Dataset):
    def __init__(self, cifar_dataset, extra_matrix):
        self.cifar_dataset = cifar_dataset
        self.extra_matrix = extra_matrix

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        extra_vector = self.extra_matrix[idx]
        return image, label, extra_vector

# çŽ°åœ¨ï¼Œåˆ›å»º CIFAR100 æ•°æ®é›†å®žä¾‹å’Œ DataLoader
cifar100_test = CIFAR100(root=os.path.expanduser("/CIFAR100"), train=False, transform=preprocess, download=True)
custom_dataset = CustomCIFAR100(cifar100_test, extra_matrix)
custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
# print(len(custom_dataloader))
# print(custom_dataloader[0])
# exit()
data_list=[]

for image, label, vector in custom_dataloader:
    data_list.append([image,label,vector])
    # print(vector.shape,image.shape,label.shape)
exit()


pass