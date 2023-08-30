"""A function to load and split the desired dataset among clients."""

def load_data(dataset_name: str, 
              dataset_path: str):
    
    assert dataset_name in ["MNIST", "CIFAR-10", "EMNIST-DIGITS", "EMNIST-BYCLASS"], f"Invalid dataset {dataset_name} requested."

    if dataset_name == "MNIST":
        from .dt_mnist import load_mnist
        return load_mnist(data_root=dataset_path)
    elif dataset_name == "EMNIST-DIGITS":
        from .dt_emnist import load_emnist
        return load_emnist(data_root=dataset_path, split="digits")
    elif dataset_name == "EMNIST-BYCLASS":
        from .dt_emnist import load_emnist
        return load_emnist(data_root=dataset_path, split="byclass")
    elif dataset_name == "CIFAR-10":
        from .dt_cifar10 import load_cifar10
        return load_cifar10(data_root=dataset_path)
    else:
        raise Exception(f"Invalid dataset {dataset_name} requested.")


def load_and_fetch_split(
        client_id: int,
        n_clients: int,
        dataset_conf: dict
    ):
    """A routine to load and split data."""

    # load the dataset requested
    trainset, testset = load_data(
        dataset_name=dataset_conf["DATASET_NAME"],
        dataset_path=dataset_conf["DATASET_PATH"],
    )
    
    distill_train, distill_test = load_data(
        dataset_name=dataset_conf["DISTILL_DATA"],
        dataset_path=dataset_conf["DATASET_PATH"],
    )

    # split the dataset if requested
    from .data_split import split_data, prepare_distill
    split_train, split_labels = split_data(
        train_data = trainset,
        dirichlet_alpha = dataset_conf["DIRICHLET_ALPHA"],
        client_id = client_id,
        n_clients = n_clients,
        random_seed = dataset_conf["RANDOM_SEED"],
    )

    # Extract specified number of samples as
    # distillation dataset from given dataset
    split_distill, split_test = None, None
    if dataset_conf["DISTILL_TEST"]:
        split_distill, split_test = prepare_distill(
            dataset = distill_test,
            num_distill = dataset_conf["DISTILL_SAMPLES"],
            random_seed = dataset_conf["RANDOM_SEED"],
        )
    else:
        split_distill, split_test = prepare_distill(
            dataset = distill_train,
            num_distill = dataset_conf["DISTILL_SAMPLES"],
            random_seed = dataset_conf["RANDOM_SEED"],
        )

    # If replacement of test set with with remaining
    # data from distill split is requested.
    if dataset_conf["REPLACE_TEST"]:
        del testset
        testset = split_test

    return (split_train, split_labels), split_distill, testset
