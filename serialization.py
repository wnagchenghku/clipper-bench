import torch
import datetime
import torchvision.models as models

trained_models = ['resnet18', 'densenet201']

def save_python_function(name):

    serialization_dir = "/tmpfs/model/{}.model".format(name)

    return serialization_dir

def deploy_pytorch_model(name,
                         pytorch_model,
                         base_image = "default"):

    try:
        serialization_dir = save_python_function(name)

        torch.save(pytorch_model, serialization_dir)

    except Exception as e:
        raise ClipperException("Error saving torch model: %s" % e)

def deploy_and_test_model(model,
                          model_name):
    deploy_pytorch_model(model_name, model)


def main():
    parser = argparse.ArgumentParser("Deploy models");
    parser.add_argument("-m", dest="model_name", type=str, required=True)
    para_sets = parser.parse_args();

    if para_sets.model_name == "all":
        for model_name in trained_models:
            deploy_and_test_model(getattr(models, model_name)(), model_name)
    else:
        deploy_and_test_model(getattr(models, para_sets.model_name)(), para_sets.model_name)

if __name__ == '__main__':
	main()