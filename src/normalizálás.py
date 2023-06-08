#csak az original képek alapján normalizáljunk!
#egyekőre az uncertaint nem veszem bele
import torch
def mean_and_std(loader):        
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)
        for images, _ in loader:
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(
          snd_moment - fst_moment ** 2)        
        return [mean,std]


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

convert_tensor=transforms.ToTensor()

dataset = datasets.ImageFolder('../input/festmenyek/', transform=convert_tensor)
loader = DataLoader(dataset)

mean_std = mean_and_std(loader)
mean_str = ''
std_str = ''
for i in range(3):
     mean_str = mean_str + str(mean_std[0][i]) + ','
     std_str = std_str + str(mean_std[1][i]) + ','

if __name__ == '__main__':
    print(dataset.imgs)
    data = open("norm.txt", "w")
    data.write(mean_str)
    data.write(std_str)
    data.close()