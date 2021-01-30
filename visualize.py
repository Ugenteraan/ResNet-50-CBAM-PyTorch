'''
Visualize the trained model's feature maps.
'''

import os
from tqdm import tqdm
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from load_dataset import LoadInputImages
from models.resnet50 import ResNet50

from runtime_args import args

print(args.use_cbam)
model_save_folder = 'resnet_cbam/' if args.use_cbam else 'resnet/'
device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
model =  ResNet50(image_depth=args.img_depth, num_classes=args.num_classes, use_cbam=args.use_cbam)

model = model.to(device)

assert os.path.exists(f"{model_save_folder}model.pth"), 'A trained model does not exist!'

try:
    state_dict = torch.load(f"{model_save_folder}model.pth", map_location=device)
    new_state_dict = OrderedDict()

    for k,v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("Model loaded!")
except Exception as e:
    print(e)

model.eval()

input_data = LoadInputImages(input_folder=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, transform=transforms.ToTensor())
data_generator = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=1)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

output_folder = './output_resnet_cbam' if args.use_cbam else './output_resnet'

if not os.path.exists(output_folder) : os.mkdir(output_folder)


fig = plt.figure(figsize=(20, 4))

for i, image in tqdm(enumerate(data_generator)):

    plt.clf()

    image = image.to(device)

    cnn_filters, output = model(image)

    #identify the predicted class
    softmaxed_output = torch.nn.Softmax(dim=1)(output)
    predicted_class = class_names[torch.argmax(softmaxed_output).cpu().numpy()]


    #merge all the filters together as one and resize them to the original image size for viewing.
    # attention_combined_filter = cv2.resize(torch.max(attention_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))
    cnn_combined_filter = cv2.resize(torch.max(cnn_filters.squeeze(0), 0)[0].detach().cpu().numpy(), (args.img_size, args.img_size))
    heatmap = np.asarray(cv2.applyColorMap(cv2.normalize(cnn_combined_filter, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                        cv2.COLORMAP_JET), dtype=np.float32)


    input_img = cv2.resize(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), (args.img_size, args.img_size))

    #create heatmap by overlaying the filters on the original image
    heatmap_cnn = cv2.addWeighted(np.asarray(input_img, dtype=np.float32), 0.9, heatmap, 0.0025, 0)

    fig.add_subplot(151)
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(152)
    plt.imshow(cnn_combined_filter)
    if args.use_cbam:
        plt.title("CNN Feature Map with CBAM")
    else:
        plt.title("CNN Feature Map without CBAM")

    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(153)
    plt.imshow(heatmap_cnn)
    plt.title("Heat Map")
    plt.xticks(())
    plt.yticks(())

    fig.suptitle(f"Network's prediction : {predicted_class.capitalize()}", fontsize=20)

    plt.savefig(f'{output_folder}/{i}.png')
