from torchvision.models import resnet

# resnet18 with 10 classes
model = resnet.resnet18(num_classes=10)