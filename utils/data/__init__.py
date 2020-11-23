"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


def get_train_dev_test_datasets(dataset, data_type, valid_size=0.1, **kwargs):
    """Return the correct instantiated train, validation, test dataset
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to load.

    data_type : {"imgs"}
        Type of dataset.
    
    valid_size : float or int, optional
        Size of the validation set. If float, should be between 0.0 and 1.0 and represent the 
        proportion of the dataset. If int, represents the absolute number of valid samples.
        0 if no validation.

    Returns
    -------
    datasets : dictionary of torch.utils.data.Dataset
        Dictionary of the `"train"`, `"valid"`, and `"valid"`.
    """

    datasets = dict()

    if data_type == "imgs":
        from .imgs import get_Dataset

    Dataset = get_Dataset(dataset)

    dataset = Dataset(split="train", **kwargs)
    if valid_size != 0:
        datasets["train"], datasets["valid"] = dataset.train_test_split(size=valid_size)
    else:
        datasets["train"], datasets["valid"] = dataset, None

    datasets["test"] = Dataset(split="test", **kwargs)

    return datasets
