'''Training script.
'''


from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from torchvision import transforms

from models.resnet50 import ResNet50
from runtime_args import args
from load_dataset import LoadDataset


device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')


train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
test_dataset = LoadDataset(dataset_folder_path=args.data_folder,image_size=args.img_size, image_depth=args.img_depth, train=False,
                            transform=transforms.ToTensor())

train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

model = ResNet50(image_depth=args.img_depth, num_classes=args.num_classes)
optimizers = Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)
summary(model, (3, 224, 224))


best_accuracy = 0
for epoch_idx in range(args.epoch):
    #Model Training & Validation.

    model.train()

    epoch_loss = []
    epoch_accuracy = []
    i = 0

    for i, sample in tqdm(enumerate(train_generator)):

        print(sample['image'].size())


