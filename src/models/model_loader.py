"""A function to load desired model for training."""

def load_model(model_configs: dict):
    
    assert model_configs["MODEL_NAME"] in ["SIMPLE-CNN", "SIMPLE-MLP", "LENET-MNIST", "LENET-CIFAR", "RESNET8-CIFAR", "RESNET18-CIFAR", "SQUEEZE-FMNIST", "SQUEEZE-CIFAR"], f"Invalid model {model_configs['MODEL_NAME']} requested."

    if model_configs["MODEL_NAME"] == "SIMPLE-MLP":
        from .simple_mlp import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "SIMPLE-CNN":
        from .simple_cnn import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "LENET-MNIST":
        from .lenet_mnist import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "LENET-CIFAR":
        from .lenet_cifar import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "RESNET8-CIFAR":
        from .resnet8_cifar import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "RESNET18-CIFAR":
        from .resnet18_cifar import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "SQUEEZE-FMNIST":
        from .squenet_fmnist import Net
        return Net 
    elif model_configs["MODEL_NAME"] == "SQUEEZE-CIFAR":
        from .squenet_cifar10 import Net
        return Net 
    else:
        raise Exception(f"Invalid model {model_configs['MODEL_NAME']} requested.")
