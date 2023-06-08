import torch
import cv2
import numpy as np
import glob as glob
import os

from model_layer import build_model
from torchvision import transforms

# Constants.
IMAGE_SIZE = 456
verzió='b5'
DEVICE = 'cpu'
DATA_PATH = '../input/rendezett_festmenyek_' + verzió +'/test/'

# Class names.
class_names = ['Bosch', 'nem-Bosch', 'other']

# Load the trained model.
model = build_model(version=verzió, pretrained=True, fine_tune=True, num_classes=2)
checkpoint = torch.load('../outputs/5layer_bigger_epoch/b5_' + str(19) + '/model_pretrained_' + verzió +'_layer.pth', map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

def mean_and_std():
    norm = open('norm.txt', 'r')
    mean_std = ''
    for row in norm:
        mean_std = row
    norm.close()
    mean_std = mean_std.split(",")
    mean_std.pop(-1)
    for i in range(len(mean_std)):
        mean_std[i] = mean_std[i][:-1]
        mean_std[i] = float(mean_std[i][7:])
    return mean_std
ms = mean_and_std()

# Get all the test image paths.
all_folders = []
for i in class_names:
    all_folders.append(DATA_PATH + i)
# Iterate over all the images and do forward pass.
all_image_paths = []
for folder in all_folders:
    all_image_paths.append(glob.glob(folder + "/*.jpeg"))

y_pred = []
y_true = []

for i in range(len(all_image_paths)):
    for j in range(len(all_image_paths[i])):
        image_path = all_image_paths[i][j]
        # Get the ground truth class name from the image path.
        gt_class_name = class_names[i]
        y_true.append(gt_class_name)
        # Read the image and create a copy.
        image = cv2.imread(image_path)
        orig_image = image.copy()
        
        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ms[0:3],
                std=ms[3:]
            )
        ])
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.to(DEVICE)
        
        # Forward pass throught the image.
        outputs = model(image)
        outputs = outputs.detach().numpy()
        pred_class_name = class_names[np.argmax(outputs[0])]
        y_pred.append(pred_class_name)
        # Annotate the image with ground truth.
        cv2.putText(
            orig_image, f"True class: {gt_class_name}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
        )
        # Annotate the image with prediction.
        cv2.putText(
            orig_image, f"Prediction: {pred_class_name}",
            (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
        ) 
        cv2.waitKey(0)
        cv2.imwrite(f"../outputs/{gt_class_name}_{str(j)}.jpeg", orig_image)

#creating a confusion matrix
import pandas
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pandas.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in class_names],
                     columns = [i for i in class_names])
plt.figure(figsize = (12,7))
seaborn.heatmap(df_cm, annot=True)
plt.savefig('../outputs/matrix_' + verzió + '_layer_big_epoch.jpeg', format='jpeg')