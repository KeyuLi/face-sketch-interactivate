from torchvision.models import vgg16

print(vgg16(pretrained=True).cuda())