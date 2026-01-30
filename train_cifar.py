import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os

from Masternet import MasterNet

def main(args):
    if args.dataset not in ["cifar10", "cifar100"]:
        raise ValueError("--dataset must be 'cifar10' or 'cifar100'")
    num_classes = 10 if args.dataset == "cifar10" else 100
    # 1. Set up data transformations and loaders
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else: # cifar100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 2. Load the network structure from the file
    print('==> Building model..')
    with open(args.arch_file, 'r') as f:
        structure_string = f.read().strip()
    
    # Use MasterNet to build the model from the string
    # Pass necessary arguments that MasterNet might expect
    model = MasterNet(num_classes=num_classes, plainnet_struct=structure_string)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if device == 'cuda':
        print("Using GPU for training.")
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    # 3. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training loop
    def train(epoch):
        print(f'\nEpoch: {epoch}')
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    # 5. Validation loop
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'==> Validation Acc: {acc:.3f}%')
        return acc

    # 6. Start training
    best_acc = 0
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        train(epoch)
        acc = test(epoch)
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f'\nEarly stopping triggered after {args.patience} epochs with no improvement.')
            break

        scheduler.step()
    
    print(f'\nTraining finished. Best validation accuracy: {best_acc:.3f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/100 Training for RZ-NAS Architectures')
    parser.add_argument('--arch_file', type=str, required=True, help='File path to the architecture string')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for training and validation')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use (cifar10 or cifar100)')
    parser.add_argument('--patience', default=10, type=int, help='How many epochs to wait for improvement before early stopping')
    args = parser.parse_args()
    main(args)

