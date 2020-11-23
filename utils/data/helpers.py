"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import numpy as np
import torch

from dib.utils.datasplit import RandomMasker
from dib.utils.helpers import tmp_seed, to_numpy


def get_masks_drop_features(drop_size, mask_shape, n_masks, n_batch=32, seed=123):
    """
    Parameters
    ----------
    drop_size : float or int or tuple, optional
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to 
        drop. If int, represents the number of datapoints to drop. If tuple, same as before 
        but give bounds (min and max). 0 means keep all.

    mask_shape : tuple of int or callable
        Shape of the mask for one example. If callable, it is given the current index.

    n_masks : int, optional
        Number of masks to return.

    n_batch : int, optional
        Size of the batches of masks => number fo concsecutive examples with the same abount of 
        kept features.

    seed : int, optional
        Random seed.

    Returns
    -------
    to_drops : list of torch.BoolTensor
        List of length n_masks where each element is a boolean tensor of shape `mask_shape` with
        1s where features should be droped.

    Examples
    --------
    >>> get_masks_drop_features(0.5, (10,), 1, n_batch=1)
    [tensor([ True, False, False, False,  True,  True, False,  True,  True, False])]
    """

    try:
        mask_shape(0)
    except TypeError:

        def mask_shape(_, ret=mask_shape):
            return ret

    if drop_size == 0:
        return [torch.zeros(1, *mask_shape(i)).bool() for i in n_batch]

    with tmp_seed(seed):
        try:
            droper = RandomMasker(
                min_nnz=drop_size[0], max_nnz=drop_size[1], is_batch_share=False
            )
        except TypeError:
            droper = RandomMasker(
                min_nnz=drop_size, max_nnz=drop_size, is_batch_share=False
            )

        to_drops = []

        for i in range(0, n_masks, n_batch):
            to_drop = droper(n_batch, mask_shape(i))
            to_drops.extend(torch.unbind(to_drop.bool(), dim=0))

    return to_drops


def get_mean_std(dataset):
    """Return the mean and std of a datset. 
    
    Examples
    --------
    >>> from .imgs import get_Dataset
    >>> import numpy as np
    >>> cifar10 = get_Dataset("cifar10")(split="test")
    Files already downloaded and verified
    >>> get_mean_std(cifar10)
    (array([0.49421427, 0.4851322 , 0.45040992], dtype=float32), array([0.24665268, 0.24289216, 0.2615922 ], dtype=float32))
    """
    dataset.rm_transformations()
    data = torch.stack([el[0] for el in dataset], dim=0)
    return np.mean(data.numpy(), axis=(0, 2, 3)), np.std(data.numpy(), axis=(0, 2, 3))


def overlay_save_datasets(
    bckgrnd_datasets,
    to_overlay_datasets,
    folder="data/",
    split_names=["train", "test"],
    dependence=dict(train=0, test=0),
    **kwargs
):
    """Overlay corresponding train and test datasetsand save the output to file.
    
    Parameters
    ----------
    bckgrnd_datasets : tuple of tuple of array like
        Background datasets on which to overlay the others. The exterior tuple corresponds to the 
        split (train, test, ...), the interior tuple is imgs/label: `((train_imgs, train_labels), ...)`, 
        image arrays should be of shape [n_bckgrnd, height_bckgrnd, width_bckgrnd, ...] and 
        dtype=uint8. Labels should be shape=[n_imgs, ...] and dtype=*.
        
    to_overlay_datasets : tuple of array like
        Datasets to overlay. Same shape and form as the previous argument `bckgrnd_datasets`.
        
    folder : str, optional
        Folder to which to save the images.
        
    split_names : list of str, optional
        Names of all the splits, should be at least as long as len(bckgrnd_datasets).

    dependence : dict of float, optional
        Whether to overlay in a way where there is dependencies between the background and the overlayed data.
        Dictionary because can be different for each split.
        
    kwargs : 
        Additional arguments to `overlay_img`.
    """
    is_missing_names = len(split_names) < len(bckgrnd_datasets)
    if is_missing_names or len(bckgrnd_datasets) != len(to_overlay_datasets):
        err = "Sizes don't agree `len(split_names)={}, len(bckgrnd_datasets)={}, len(to_overlay_datasets)={}`."
        raise ValueError(
            err.format(
                len(split_names), len(bckgrnd_datasets), len(
                    to_overlay_datasets)
            )
        )

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, (bckgrnd, to_overlay, name) in enumerate(
        zip(bckgrnd_datasets, to_overlay_datasets, split_names)
    ):
        if to_overlay[0] is not None and bckgrnd[0] is not None:
            if dependence is not None and dependence[name] != 0:
                out, idcs = overlay_img_dependencies(
                    bckgrnd[0],
                    to_overlay[0],
                    bckgrnd[1],
                    to_overlay[1],
                    dependence=dependence[name],
                    **kwargs
                )
            else:
                # no dependecies between bckground and overlay
                out, idcs = overlay_img(bckgrnd[0], to_overlay[0], **kwargs)

            np.save(os.path.join(folder, name + "_x.npy"),
                    out, allow_pickle=False)
            np.save(
                os.path.join(folder, name + "_y.npy"),
                to_numpy(bckgrnd[1]),
                allow_pickle=False,
            )
            np.save(
                os.path.join(folder, name + "_y_distractor.npy"),
                to_numpy(to_overlay[1])[idcs],
                allow_pickle=False,
            )


def overlay_img_dependencies(
    bckgrnd,
    to_overlay,
    bckgrnd_labels,
    to_overlay_labels,
    dependence=0.5,
    seed=123,
    **kwargs
):
    """Overlays an image `to_overlay` on a `bckgrnd` with dependencies between the labels.

    Parameters
    ----------
    bckgrnd : array like, shape=[n_bckgrnd, height_bckgrnd, width_bckgrnd, ...], dtype=uint8
        Background images. Each image will have one random image from  `to_overlay` overlayed on it.
    
    to_overlay : array like, shape=[n_overlay, height_overlay, width_overlay, ...], dtype=uint8
        Images to overlay. Currently the following assumptions are made:
            - the overlaid images have to be at most as big as the background ones (i.e. 
              `height_bckgrnd <= height_bckgrnd` and `<= width_bckgrnd`).
            - The overlayed images are also used as mask. This is especially good for black 
              and white images : whiter pixels (~1) are the ones to be overlayed. In the case
              of colored image, this still hold but channel wise.

    bckgrnd_labels : array like, shape=[n_bckgrnd]
        Labels of the background images. 

    to_overlay_labels : array like, shape=[n_overlay]
        Labels of the images to overlay. The number of unique labels need to be larger than the unique
        labels of background images.

    dependence : float, optional
        Level of positive dependence in [0,1]. If 0 no dependence. If 1 then label of overlayed
        is the same as label of background all the time.
        
    seed : int, optional
        Pseudo random seed.

    kwargs :
        Additional arguments to `overlay_img`.
    """
    bckgrnd_labels = to_numpy(bckgrnd_labels)
    to_overlay_labels = to_numpy(to_overlay_labels)
    bckgrnd = to_numpy(bckgrnd)
    to_overlay = to_numpy(to_overlay)

    out_imgs = np.zeros_like(bckgrnd)
    out_idcs = np.zeros_like(bckgrnd_labels)

    for i in np.unique(bckgrnd_labels):
        bckgrnd_i = bckgrnd[bckgrnd_labels == i]
        to_overlay_i = to_overlay[to_overlay_labels == i]
        n_bckgrnd_i = bckgrnd_i.shape[0]
        n_overlay_i = to_overlay_i.shape[0]

        with tmp_seed(seed):
            n_dependent = int(dependence * n_bckgrnd_i)
            idx_to_overlay_i = np.random.choice(
                range(n_overlay_i), size=n_dependent)
            to_overlay_i = to_overlay_i[idx_to_overlay_i]

        out, idcs = overlay_img(
            bckgrnd_i[:n_dependent], to_overlay_i, seed=seed, **kwargs
        )
        # indices in terms of the actual indices
        idcs = idx_to_overlay_i[idcs]
        idcs = np.flatnonzero(to_overlay_labels == i)[idcs]

        if n_dependent < n_bckgrnd_i:

            with tmp_seed(seed):
                # sampling without dependency => from the entire set
                n_independent = n_bckgrnd_i - n_dependent
                idx_to_overlay_i = np.random.choice(
                    range(to_overlay.shape[0]), size=n_independent
                )

            out_indep, idcs_indep = overlay_img(
                bckgrnd_i[n_dependent:],
                to_overlay[idx_to_overlay_i],
                seed=seed,
                **kwargs
            )
            # indices in terms of the actual indices
            idcs_indep = idx_to_overlay_i[idcs_indep]

            out = np.concatenate([out, out_indep])
            idcs = np.concatenate([idcs, idcs_indep])

        # put it in order compared to initial order of background
        out_imgs[bckgrnd_labels == i] = out
        out_idcs[bckgrnd_labels == i] = idcs

    return out_imgs, out_idcs


def overlay_img(bckgrnd, to_overlay, is_shift=False, seed=123):
    """Overlays an image with black background `to_overlay` on a `bckgrnd`
    
    Parameters
    ----------
    bckgrnd : array like, shape=[n_bckgrnd, height_bckgrnd, width_bckgrnd, ...], dtype=uint8
        Background images. Each image will have one random image from  `to_overlay` overlayed on it.
    
    to_overlay : array like, shape=[n_overlay, height_overlay, width_overlay, ...], dtype=uint8
        Images to overlay. Currently the following assumptions are made:
            - the overlaid images have to be at most as big as the background ones (i.e. 
              `height_bckgrnd <= height_bckgrnd` and `<= width_bckgrnd`).
            - The overlayed images are also used as mask. This is especially good for black 
              and white images : whiter pixels (~1) are the ones to be overlayed. In the case
              of colored image, this still hold but channel wise.

    is_shift : bool, optional
        Whether to randomly shift all overlayed images or to keep them on the bottom right.

        
    seed : int, optional
        Pseudo random seed.
        
    Return
    ------
    imgs : np.array, shape=[n_bckgrnd, height, width, 3], dtype=uint8
        Overlayed images.
        
    selected : np.array, shape=[n_bckgrnd], dtype=int64
        Indices of the selected overlayed images.
    """
    bckgrnd = to_numpy(bckgrnd)
    to_overlay = to_numpy(to_overlay)

    with tmp_seed(seed):
        n_bckgrnd = bckgrnd.shape[0]
        n_overlay = to_overlay.shape[0]
        selected = np.random.choice(np.arange(n_overlay), size=n_bckgrnd)
        to_overlay = to_overlay[selected, ...]

    bckgrnd = ensure_color(bckgrnd).astype(np.float32)
    to_overlay = ensure_color(to_overlay).astype(np.float32)
    over_shape = to_overlay.shape[1:]
    bck_shape = bckgrnd.shape[1:]

    def get_margin(i): return (bck_shape[i] - over_shape[i]) // 2
    def get_max_shift(i): return get_margin(i) + over_shape[i] // 3
    get_shift = (
        lambda i: np.random.randint(-get_max_shift(i), get_max_shift(i))
        if is_shift
        else get_max_shift(i) // 2
    )

    resized_overlay = np.zeros((n_bckgrnd,) + bck_shape[:2] + over_shape[2:])
    resized_overlay[
        :,
        get_margin(0): -get_margin(0) or None,
        get_margin(1): -get_margin(1) or None,
    ] = to_overlay

    for i in range(2):  # shift x and y
        resized_overlay = np.stack(
            [np.roll(im, get_shift(i), axis=i) for im in resized_overlay]
        )

    mask = resized_overlay / 255

    return (mask * resized_overlay + (1 - mask) * bckgrnd).astype(np.uint8), selected


def at_least_ndim(arr, ndim):
    """Ensures that a numpy array is at least `ndim`-dimensional."""
    padded_shape = arr.shape + (1,) * (ndim - len(arr.shape))
    return arr.reshape(padded_shape)


def ensure_color(imgs):
    """
    Ensures that a batch of colored (3 channels) or black and white (1 channels) numpy uint 8 images 
    is colored (3 channels).
    """
    imgs = at_least_ndim(imgs, 4)
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)
    return imgs


def bw_to_color(img, rgb_proba=[1, 0, 0], seed=None):
    """Transform black and white image to red green or blue with a given probability."""
    with tmp_seed(seed):
        channel_to_color = np.random.choice(3, size=1, p=rgb_proba)

    frame = torch.zeros_like(img.expand(3, -1, -1))
    frame[channel_to_color] = img
    return frame
