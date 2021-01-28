'''Training script.
'''


import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from torchvision import transforms

from models.resnet50 import ResNet50
from runtime_args import args
from load_dataset import LoadDataset
from plot import plot_loss_acc


device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

if not os.path.exists(args.graphs_folder) : os.mkdir(args.graphs_folder)

train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
                            transform=transforms.ToTensor())

train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

model = ResNet50(image_depth=args.img_depth, num_classes=args.num_classes, use_cbam=args.use_cbam)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)
summary(model, (3, 224, 224))


training_loss_list = []
training_acc_list = []
testing_loss_list = []
testing_acc_list = []

best_accuracy = 0
for epoch_idx in range(args.epoch):
    #Model Training & Validation.
    model.train()

    epoch_loss = []
    epoch_accuracy = []
    i = 0

    for i, sample in tqdm(enumerate(train_generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        optimizer.zero_grad()

        net_output = model(batch_x)
        total_loss = criterion(input=net_output, target=batch_y)

        total_loss.backward()
        optimizer.step()
        batch_accuracy = model.calculate_accuracy(predicted=net_output, target=batch_y)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_accuracy)

    curr_accuracy = sum(epoch_accuracy)/(i+1)
    curr_loss = sum(epoch_loss)/(i+1)

    training_loss_list.append(curr_accuracy)
    training_acc_list.append(curr_loss)

    print(f"Epoch {epoch_idx}")
    print(f"Training Loss : {curr_loss}, Training accuracy : {curr_accuracy}")

    model.eval()
    epoch_loss = []
    epoch_accuracy = []
    i = 0

    with torch.set_grad_enabled(False):
        for i, sample in tqdm(enumerate(test_generator)):

            batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

            net_output = model(batch_x)

            total_loss = criterion(input=net_output, target=batch_y)

            batch_accuracy = model.calculate_accuracy(predicted=net_output, target=batch_y)
            epoch_loss.append(total_loss.item())
            epoch_accuracy.append(batch_accuracy)

        curr_accuracy = sum(epoch_accuracy)/(i+1)
        curr_loss = sum(epoch_loss)/(i+1)

        testing_loss_list.append(curr_accuracy)
        testing_acc_list.append(curr_loss)

    if epoch_idx % 5 == 0:
        plot_loss_acc(path=args.graphs_folder, num_epoch=epoch_idx, train_accuracies=training_acc_list, train_losses=training_loss_list,
                          test_accuracies=testing_acc_list, test_losses=testing_loss_list)

        lr_decay.step() #decrease the learning rate at every n epoch.

    print(f"Testing Loss : {curr_loss}, Testing accuracy : {curr_accuracy}")
    print('--------------------------------------------------------------------------------')