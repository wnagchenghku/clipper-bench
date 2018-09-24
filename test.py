import torchvision.models as models
import datetime
import torch

torch.manual_seed(0)
model = models.resnet18()

input = torch.FloatTensor(1, 3, 224, 224) # allocates
input.fill_(1) # fills with 1

X = torch.autograd.Variable(input)
predict_start = datetime.datetime.now() = datetime.datetime.now()
model(X) # runs the model
predict_end = datetime.datetime.now() = datetime.datetime.now()

print("prediction takes %d s, %d us\n" % ((predict_end - predict_start).seconds, (predict_end - predict_start).microseconds))
