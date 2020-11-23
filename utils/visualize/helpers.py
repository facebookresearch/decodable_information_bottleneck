"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np


# example : https://github.com/matplotlib/matplotlib/issues/7008
def kwargs_log_xscale(x_data, mode="equidistant", base=None):
    """Return arguments to set log_scale as one would wish. mode=["smooth","equidistant"]."""

    # if constant diff don't use logscale
    if base == 1 or np.diff(x_data).var() == 0:
        return dict(value="linear")

    # automatically compute base
    if base is None:
        # take avg multiplier between each consecutive elements as base i.e 2,8,32 would be 4
        # but 0.1,1,10 would be 10
        base = int((x_data[x_data > 0][1:] /
                    x_data[x_data > 0][:-1]).mean().round())

    if (x_data <= 0).any():
        min_nnz_x = np.abs(x_data[x_data != 0]).min()
        if mode == "smooth":
            linscalex = np.log(np.e) / np.log(base) * (1 - (1 / base))
        elif mode == "equidistant":
            linscalex = 1 - (1 / base)
        else:
            raise ValueError(f"Unkown mode={mode}")

        return dict(
            value="symlog",
            linthreshx=min_nnz_x,
            basex=base,
            subsx=list(range(base)),
            linscalex=linscalex,
        )
    else:
        return dict(value="log", basex=base, subsx=list(range(base)))
