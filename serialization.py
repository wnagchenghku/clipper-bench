import torch
import datetime
import torchvision.models as models

PATH = "/tmpfs/model/resnet.model"

def main():

	torch.save(models.resnet18(), PATH)

	load_start = datetime.datetime.now()

	model = torch.load(PATH)

	load_end = datetime.datetime.now()

	print("model loading takes %d s, %d us\n" % ((load_end - load_start).seconds, (load_end - load_start).microseconds))

	input = torch.randn(1, 3, 224, 224)

	output = model(input)

	print(output)

if __name__ == '__main__':
	main()