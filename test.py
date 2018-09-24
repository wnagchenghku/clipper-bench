import torchvision.models as models
import datetime
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse model name.')
    parser.add_argument('-m', type=str, dest="model_name", action="store", required=True)
    args = parser.parse_args()

    serialization_dir = "/tmpfs/model/{}.model".format(args.model_name)
    model = getattr(models, args.model_name)()
    load_start = datetime.datetime.now()
    model.load_state_dict(torch.load(serialization_dir)) # loads only the model parameters
    load_end = datetime.datetime.now()
    inputs.FloatTensor(1, 3, 224, 224)
    inputs.fill_(1)
    predict_start = datetime.datetime.now()
    model(torch.autograd.Variable(inputs))
    predict_end = datetime.datetime.now()
    print("prediction takes %d s, %d us\n" % ((predict_end - predict_start).seconds, (predict_end - predict_start).microseconds))
    print("model loading takes %d s, %d us\n" % ((load_end - load_start).seconds, (load_end - load_start).microseconds))
    