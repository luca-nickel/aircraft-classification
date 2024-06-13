import torch
from dataset import get_dataloaders
from model import CNN
from pathlib import Path

data_root = Path("../../data/fgvc-aircraft-2013b")

trainloader, testloader, num_classes = get_dataloaders(data_root, batch_size=4, num_workers=2)
net = CNN(num_classes=num_classes)

# Laden des trainierten Modells
net.load_state_dict(torch.load('model.pth'))

# Evaluierung des Modells auf dem Testdatensatz
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Genauigkeit des Netzwerks auf den Testbildern: %d %%' % (
    100 * correct / total))
