####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
# - look at the effect of Q minimality on different tasks : training loss, random label loss, mnist, worst case test loss, test loss

experiment="dev"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar10mnist
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-5
train.optim=adam
train.monitor_best=tloss 
model.architecture.z_dim=1024
hydra.launcher.time=1000
model.Q_zy.hidden_size=1024
model.Q_zy.n_hidden_layers=1
datasize.max_epochs=50
model.loss.altern_minimax=0
model.loss.n_per_head=1
clfs=default
datasize=all
model=cdib
is_nvidia_smi=True
$dev
"



if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

    wait 


    python main.py $kwargs $kwargs_multi $kwargs1 -m &
      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi
