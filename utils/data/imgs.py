"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from dib.utils.helpers import to_numpy

from .base import BaseDataset
from .helpers import bw_to_color, get_masks_drop_features, overlay_save_datasets

COLOUR_BLACK = torch.tensor([0.0, 0.0, 0.0])
COLOUR_WHITE = torch.tensor([1.0, 1.0, 1.0])
COLOUR_BLUE = torch.tensor([0.0, 0.0, 1.0])
DATASETS_DICT = {
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "bincifar100": "BinaryCIFAR100",
    "binsvhn": "BinarySVHN",
    "binmnist": "BinaryMNIST",
    "coloredmnist": "ColoredMNIST",
    "svhn": "SVHN",
    "bincifar10mnist": "BinCifar10Mnist",
    "cifar10mnist": "Cifar10Mnist",
    "cifar10mnistshift": "Cifar10MnistShift",
    "bincifar10mnistdep9": "BinCifar10MnistDep9",
    "bincifar10mnistdep8": "BinCifar10MnistDep8",
    "bincifar10mnistdep7": "BinCifar10MnistDep7",
    "bincifar10mnistdep5": "BinCifar10MnistDep5",
    "bincifar10mnistdep3": "BinCifar10MnistDep3",
    "cifar10mnistdep9": "Cifar10MnistDep9",
    "cifar10mnistdep8": "Cifar10MnistDep8",
    "cifar10mnistdep7": "Cifar10MnistDep7",
    "cifar10mnistdep5": "Cifar10MnistDep5",
    "cifar10mnistdep3": "Cifar10MnistDep3",
}
DATASETS = list(DATASETS_DICT.keys())


logger = logging.getLogger(__name__)


# HELPERS
def get_Dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_Dataset(dataset).shape


class ImgDataset(BaseDataset):
    """Image dataset wrapper that adds nice functionalitites.
    
    Parameters
    ----------
    is_augment : bool, optional
        Whether to transform the training set (and thus validation).

    split : {'train', 'test', ...}, optional
        According dataset is selected.

    translation : int or sequence of int, optional
        Max translation of the image translate the images when using data augmentation.

    is_flip : int, optional
        Whether to apply horizontal flipping. Only applied if not a number

    rotation : float or sequence of float, optional
        Range of degrees to select from.

    is_normalize : bool, optional   
        Whether to normalize the dataset.

    target_transform : callable, optional
        Transformation of the target.
    """

    is_numbers = False

    def __init__(
        self,
        *args,
        is_augment=True,
        split="train",
        translation=4,
        is_flip=True,
        rotation=15,
        is_normalize=True,
        target_transform=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.is_augment = is_augment
        self.split = split
        self.is_drop_features = False  # by default return all features
        self.translation = translation
        self.is_flip = is_flip and not self.is_numbers
        self.rotation = rotation
        self.is_normalize = is_normalize
        self.target_transform = target_transform

        if self.is_augment and self.split == "train":
            self.transform = transforms.Compose(self.get_train_transforms())
        else:
            self.transform = transforms.Compose(self.get_test_transforms())

    def get_train_transforms(self):
        """Return the training transformation."""
        return [
            transforms.Resize((self.shape[1], self.shape[2])),
            # the following performs translation
            transforms.RandomCrop(
                (self.shape[1], self.shape[2]), padding=self.translation
            ),
            # don't flip if working with numbers
            transforms.RandomHorizontalFlip() if self.is_flip else torch.nn.Identity(),
            transforms.RandomRotation(self.rotation),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            if self.is_normalize
            else torch.nn.Identity(),
        ]

    def get_test_transforms(self):
        """Return the testing transformation."""
        return [
            transforms.Resize((self.shape[1], self.shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            if self.is_normalize
            else torch.nn.Identity(),
        ]

    def rm_transformations(self):
        """Completely remove transformation. Used to plot or compute mean and variance."""
        self.transform = transforms.Compose([transforms.ToTensor()])

    def make_test(self):
        """Make the data a test set."""
        self.transform = transforms.Compose(self.get_test_transforms())

    def drop_features_(self, drop_size):
        """Drop part of the features (pixels in images).

        Note
        ----
        - this function actually just precomputes the `self.to_drop` of values that should be droped 
        the dropping is in `__get_item__`.

        Parameters
        ----------
        drop_size : float or int or tuple, optional
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
            drop. If int, represents the number of datapoints to drop. If tuple, same as before 
            but give bounds (min and max). 0 means keep all.
        """
        self.logger.info(f"drop_features_ {drop_size} features...")

        assert not self.is_drop_features, "cannot drop multiple times the features"

        self.is_drop_features = True

        self.to_drop = get_masks_drop_features(
            drop_size, [self.shape[1], self.shape[2]], len(self), seed=self.seed
        )

    def __getitem__(self, index):

        if self.targets.ndim > 1:
            multi_target = self.targets
            # often datasets have code that can only deal with a single target
            self.targets = multi_target[:, 0]
            X, target = super().__getitem__(index)
            self.targets = multi_target  # set back multi targets
            multi_target = (target,) + tuple(self.targets[index, 1:])
        else:
            X, target = super().__getitem__(index)
            multi_target = (target,)

        multi_target = self.add_index(multi_target, index)

        if self.is_drop_features:
            X[:, self.to_drop[index]] = float("nan")

        return X, multi_target


# TORCHVISION DATASETS
class SVHN(ImgDataset, datasets.SVHN):
    """SVHN wrapper. Docs: `datasets.SVHN.`

    Parameters
    ----------
    kwargs:
        Additional arguments to `ImgDataset`.

    Examples
    --------
    >>> data = SVHN(split="train") #doctest:+ELLIPSIS
    Using ...
    >>> len(data)
    73257
    >>> len(data) == len(data.data) == len(data.targets) 
    True
    >>> [type(i) for i in data[0]]
    [<class 'torch.Tensor'>, <class 'int'>]

    >>> from .helpers import get_mean_std
    >>> mean, std = get_mean_std(data)
    >>> (str(list(mean)) == str(data.mean)) and (str(list(std)) == str(data.std))
    True

    >>> train, valid = data.train_test_split(size=1000, is_stratify=True)
    >>> len(valid)
    1000

    >>> data.drop_labels_(0.9)
    >>> round(len([t for t in data.targets if t == -1]) / len(data), 1)
    0.9

    >>> data.balance_labels_()
    >>> len(data)
    131864

    >>> data.drop_unlabelled_()
    >>> len(data)
    65932

    >>> data.drop_features_(0.7)
    >>> round((torch.isnan(data[0][0])).float().mean().item(), 1)
    0.7
    >>> data.set_test_transforms() # for replicability
    >>> data[0][0][0] # showing image for one channel
    tensor([[    nan,     nan,  0.2850,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan,  0.2652,     nan,  ...,     nan,     nan,     nan],
            ...,
            [ 0.1067,     nan,  0.1860,  ..., -0.4477,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,     nan,     nan],
            [    nan,     nan,     nan,  ...,     nan,  0.1067,     nan]])

    >>> data[0][1]
    1
    >>> data.randomize_targets_()
    >>> data[0][1]
    8
    """

    shape = (3, 32, 32)
    missing_px_color = COLOUR_BLACK
    n_classes = 10
    n_train = 73257
    mean = [0.43768448, 0.4437684, 0.4728041]
    std = [0.19803017, 0.20101567, 0.19703583]
    is_numbers = True

    def __init__(self, **kwargs):
        ImgDataset.__init__(self, **kwargs)
        datasets.SVHN.__init__(
            self,
            self.root,
            download=True,
            split=self.split,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.labels = to_numpy(self.labels)

        if self.is_random_targets:
            self.randomize_targets_()

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels

    @targets.setter
    def targets(self, values):
        self.labels = values


class CIFAR10(ImgDataset, datasets.CIFAR10):
    """CIFAR10 wrapper. Docs: `datasets.CIFAR10.`

    Parameters
    ----------
    kwargs:
        Additional arguments to `datasets.CIFAR10` and `ImgDataset`.

    Examples
    --------
    See SVHN for more examples.

    >>> data = CIFAR10(split="train") #doctest:+ELLIPSIS
    Files ...
    >>> from .helpers import get_mean_std
    >>> mean, std = get_mean_std(data)
    >>> list(std)
    [0.24703279, 0.24348423, 0.26158753]
    >>> (str(list(mean)) == str(data.mean)) and (str(list(std)) == str(data.std))
    True
    """

    shape = (3, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLACK
    n_train = 50000
    mean = [0.4914009, 0.48215896, 0.4465308]
    std = [0.24703279, 0.24348423, 0.26158753]

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.CIFAR10.__init__(
            self,
            self.root,
            download=True,
            train=self.split == "train",
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.targets = to_numpy(self.targets)

        if self.is_random_targets:
            self.randomize_targets_()


class CIFAR100(ImgDataset, datasets.CIFAR100):
    """CIFAR100 wrapper. Docs: `datasets.CIFAR100.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test'}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR100` and `ImgDataset`.

    Examples
    --------
    See SVHN for more examples.

    >>> data = CIFAR100(split="train") #doctest:+ELLIPSIS
    Files ...
    >>> from .helpers import get_mean_std
    >>> mean, std = get_mean_std(data)
    >>> (str(list(mean)) == str(data.mean)) and (str(list(std)) == str(data.std))
    True
    """

    shape = (3, 32, 32)
    n_classes = 100
    n_train = 50000
    missing_px_color = COLOUR_BLACK
    mean = [0.5070754, 0.48655024, 0.44091907]
    std = [0.26733398, 0.25643876, 0.2761503]

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.CIFAR100.__init__(
            self,
            self.root,
            download=True,
            train=self.split == "train",
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.targets = to_numpy(self.targets)

        if self.is_random_targets:
            self.randomize_targets_()


class MNIST(ImgDataset, datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST` and `ImgDataset`.

    Examples
    --------
    See SVHN for more examples.

    >>> data = MNIST(split="train") 
    >>> from .helpers import get_mean_std
    >>> mean, std = get_mean_std(data)
    >>> (str(list(mean)) == str(data.mean)) and (str(list(std)) == str(data.std))
    True
    """

    shape = (1, 32, 32)
    n_classes = 10
    n_examples = 60000
    missing_px_color = COLOUR_BLUE
    mean = [0.13066062]
    std = [0.30810776]
    is_numbers = True

    def __init__(self, **kwargs):

        ImgDataset.__init__(self, **kwargs)
        datasets.MNIST.__init__(
            self,
            self.root,
            download=True,
            train=self.split == "train",
            transform=self.transform,
            target_transform=self.target_transform,
        )
        self.targets = to_numpy(self.targets)

        if self.is_random_targets:
            self.randomize_targets_()

    def append_(self, other):
        """Append a dataset to the current one."""
        # mnist data is in torch format
        self.data = torch.cat([self.data, other.data], dim=0)
        self.targets = np.append(self.targets, other.targets, axis=0)


class ColoredMNIST(MNIST):
    """Colored binary MNIST where the color is a noisy predictor of the label. Binary label is larger
    or smaller than 10.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected. 

    noise : float, optional
        Probability of being wrong when only considering the color.

    is_fixed : bool, optional
        Whether to resample colors at every epoch though the noise.

    is_wrong_test : bool, optional
        Whether to use a test set where color is useless (wrong 100 %) of the time.

    kwargs:
        Additional arguments to `datasets.MNIST` and `ImgDataset`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> n = 2000
    >>> data = ColoredMNIST(split="train", noise=0.2)
    >>> df = pd.DataFrame([(data[i][1][0], int(data[i][0][0].sum() == 0)) for i in range(n)], columns = ["lab","col"]) 
    >>> out = df.groupby(["lab","col"]).size().values / n
    >>> (np.array([0.37,0.07,0.07,0.37]) < out).all() and (out < np.array([0.43,0.13,0.13,0.43])).all()
    True
    >>> data = ColoredMNIST(split="test", noise=0.2, is_noisy_test=True)
    >>> df = pd.DataFrame([(data[i][1][0], int(data[i][0][0].sum() == 0)) for i in range(n)], columns = ["lab","col"]) 
    >>> out = df.groupby(["lab","col"]).size().values / n
    >>> (np.array([0.00,0.47,0.47,0.00]) <= out).all() and (out < np.array([0.08,0.48,0.48,0.08])).all()
    True
    """

    shape = (3, 32, 32)
    n_classes = 2
    n_examples = 60000

    def __init__(self, noise=0.1, is_fixed=True, is_noisy_test=True, **kwargs):
        super().__init__(**kwargs)
        self.targets = (self.targets < 5).astype(int)
        self.is_fixed = is_fixed
        self.noise = noise
        self.is_noisy_test = is_noisy_test

        if self.is_noisy_test and self.split == "test":
            self.noise = 1

    @property
    def raw_folder(self):
        # use mnist data
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self):
        # use mnist data
        return os.path.join(self.root, "MNIST", "processed")

    def __getitem__(self, index):
        X, multi_target = super().__getitem__(index)

        # by using the index for the seed ensures that same across epochs
        seed = index if self.is_fixed else None

        if multi_target[0] == 0:
            X = bw_to_color(
                X, rgb_proba=[1 - self.noise, self.noise, 0], seed=seed)
        else:
            X = bw_to_color(
                X, rgb_proba=[self.noise, 1 - self.noise, 0], seed=seed)

        return X, multi_target


# OTHER DATASETS
class BinaryCIFAR100(CIFAR100):
    """Like CIFAR100 but 2 class (odd or even)."""

    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


class BinarySVHN(SVHN):
    """Like SVHN but 2 class (odd or even)."""

    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


class BinaryMNIST(MNIST):
    """Like MNIST but 2 class (odd or even)."""

    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2

    @property
    def raw_folder(self):
        # use mnist data
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self):
        # use mnist data
        return os.path.join(self.root, "MNIST", "processed")


class OverlayedDatasetBase(ImgDataset, Dataset):
    """Overlays 2 other datasets.
    
    Note
    ----
    - Randomization of targets / stratification / ... Will still be done on the actual 
    targets instead of the distractor.
    - only overlaying train and test and not other splits.
    - The second value of the target will be the distractor. Index is always last.
    - This a base class that should be inherited from. The only required changed are
    adding class generic attributed such as Overlayed, Background, shift. 
    See Cifar10Mnist for an example.

    Parameters. 
    ----------
    kwargs : 
        Additional arguments to ImgDataset.
    
    Attributes
    ----------
    Background : ImgDataset
        Dataset to use as background. 
        
    Overlayed : ImgDataset
        Dataset to overlay on the background. Currently the following assumptions are made:
        - the overlaid images have to be at most as big as the background ones (i.e. 
          `height_bckgrnd <= height_bckgrnd` and `<= width_bckgrnd`).
        - The overlayed images are also used as mask. This is especially good for black 
          and white images : whiter pixels (~1) are the ones to be overlayed. In the case
          of colored image, this still hold but channel wise.

    dependence : dict of float, optional
        Whether to overlay in a way where there is dependencies between the background and the overlayed data.
        Dictionary because can be different for each split.
    """

    add_dist = True  # whether to add the distractor to the target
    named_dir = None
    # Whether to randomly shift all overlayed images or to keep them on the bottom right.
    is_shift = False
    dependence = dict(train=0, test=0)

    def __init__(self, **kwargs):
        ImgDataset.__init__(self, **kwargs)

        name = self.named_dir if self.named_dir is not None else type(
            self).__name__
        self.dir = os.path.join(self.root, name)

        if not os.path.isdir(self.dir):
            self.make_dataset()

        self.data = np.load(os.path.join(self.dir, f"{self.split}_x.npy"))
        self.targets = np.load(os.path.join(self.dir, f"{self.split}_y.npy"))
        self.distractor = np.load(
            os.path.join(self.dir, f"{self.split}_y_distractor.npy")
        )

        if self.is_random_targets:
            self.randomize_targets_()  # doesn't randomize distractors

    def __len__(self):
        return len(self.targets)

    @property
    def map_target_position(self):
        """
        Return a dictionary that maps the type of target (e.g. "index") to its position in the 
        outputted target.
        """
        map_target_position = super().map_target_position

        if self.add_dist:
            map_target_position["distractor"] = len(map_target_position)

        return map_target_position

    def add_distractor(self, y, index):
        """Append the distractor to the targets."""
        if self.add_dist:
            try:
                y = tuple(y) + (self.distractor[index],)
            except TypeError:
                y = [y, self.distractor[index]]
        return y

    def _switch_distractor_target(self):
        """Switch the distractor and the target."""
        self.targets, self.distractor = self.distractor, self.targets

    def keep_indcs_(self, indcs):
        super().keep_indcs_(indcs)
        self.distractor = self.distractor[indcs]

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # make sure it's a tuple
        try:
            target = tuple(target)
        except TypeError:
            target = (target,)

        target = self.add_index(target, index)
        target = self.add_distractor(target, index)

        if self.is_drop_features:
            img[:, self.to_drop[index]] = float("nan")

        return img, target

    def make_dataset(self):
        logger.info("Overlaying the datasets...")
        overlay_train = self.Overlayed()
        bckgrnd_train = self.Background()

        overlay_train.rm_transformations()
        bckgrnd_train.rm_transformations()

        overlay_test = self.Overlayed(split="test")
        bckgrnd_test = self.Background(split="test")

        overlay_test.rm_transformations()
        bckgrnd_test.rm_transformations()

        bckgrnd_datasets = (
            [bckgrnd_train.data, bckgrnd_train.targets],
            [bckgrnd_test.data, bckgrnd_test.targets],
        )

        overlay_datasets = (
            [overlay_train.data, overlay_train.targets],
            [overlay_test.data, overlay_test.targets],
        )

        overlay_save_datasets(
            bckgrnd_datasets,
            overlay_datasets,
            folder=self.dir,
            is_shift=self.is_shift,
            dependence=self.dependence,
        )

    def append_(self, other):
        super().append_(other)
        self.distractor = np.append(self.distractor, other.distractor, axis=0)


class Cifar10Mnist(OverlayedDatasetBase):
    shape = (3, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLACK
    n_train = 50000
    mean = [0.52897614, 0.5220055, 0.49050677]
    std = [0.2650898, 0.263235, 0.28332546]
    Overlayed = MNIST
    Background = CIFAR10
    is_shift = False


class Cifar10MnistShift(Cifar10Mnist):
    is_shift = True


class BinCifar10Mnist(Cifar10Mnist):
    """Like Cifar10Mnist but 2 class (odd or even)."""

    named_dir = "Cifar10Mnist"  # same dataset as Cifar10Mnist => don't recreate
    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


# Correlated overlayed


class Cifar10MnistDep9(Cifar10Mnist):
    """Cifar10mnist where the mnist is correlated with cifar10."""

    dependence = dict(train=0.9, test=0)


class Cifar10MnistDep8(Cifar10Mnist):
    """Cifar10mnist where the mnist is correlated with cifar10."""

    dependence = dict(train=0.8, test=0)


class Cifar10MnistDep7(Cifar10Mnist):
    """Cifar10mnist where the mnist is correlated with cifar10."""

    dependence = dict(train=0.7, test=0)


class Cifar10MnistDep5(Cifar10Mnist):
    """Cifar10mnist where the mnist is correlated with cifar10."""

    dependence = dict(train=0.5, test=0)


class Cifar10MnistDep3(Cifar10Mnist):
    """Cifar10mnist where the mnist is correlated with cifar10."""

    dependence = dict(train=0.3, test=0)


class BinCifar10MnistDep9(Cifar10MnistDep9):
    """Like Cifar10Mnist but 2 class (odd or even)."""

    named_dir = "Cifar10MnistDep9"  # same dataset as Cifar10Mnist => don't recreate
    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


class BinCifar10MnistDep8(Cifar10MnistDep8):
    """Like Cifar10Mnist but 2 class (odd or even)."""

    named_dir = "Cifar10MnistDep8"  # same dataset as Cifar10Mnist => don't recreate
    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


class BinCifar10MnistDep7(Cifar10MnistDep7):
    """Like Cifar10Mnist but 2 class (odd or even)."""

    named_dir = "Cifar10MnistDep7"  # same dataset as Cifar10Mnist => don't recreate
    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


class BinCifar10MnistDep5(Cifar10MnistDep5):
    """Like Cifar10Mnist but 2 class (odd or even)."""

    named_dir = "Cifar10MnistDep5"  # same dataset as Cifar10Mnist => don't recreate
    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2


class BinCifar10MnistDep3(Cifar10MnistDep3):
    """Like Cifar10Mnist but 2 class (odd or even)."""

    named_dir = "Cifar10MnistDep3"  # same dataset as Cifar10Mnist => don't recreate
    n_classes = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.targets = self.targets % 2
