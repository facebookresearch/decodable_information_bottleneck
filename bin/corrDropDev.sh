####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether dropout can be seen as decreasing the decodable mutual information. Note that no dropout on zx heads
# Hypothesis
#   - The generalization gap will get smaller with larger dropout, and the DI[Z->X] will get smaller.

experiment="corrDropDev"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_augment=False
model.architecture.z_dim=1024
clfs=default
train.clf_kwargs.clean_after_run=training
train.kwargs.lr=5e-5
train.monitor_best=last
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=2
train.trnsf_kwargs.max_epochs=300
train.clf_kwargs.max_epochs=100
hydra.launcher.time=300
model=erm
datasize=all
is_correlation=True
is_correlation_Bob=True
clfs.is_reinitialize=False
is_skip_clf_if_precomputed=False
train.clf_kwargs.is_train=False
$dev
"

kwargs_multi="
run=0,1,2,3,4
dataset=cifar10,svhn
encoder=mlp,resnet18
model.dropout=0.,0.2,0.4,0.6,0.8,.9
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # precompute the transformer if not already done
    python main.py $kwargs $kwargs_multi $kwargs1 -m &
      
  done
fi

wait 


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.col=encoder \
       plot_generalization.x=dropout \
       plot_generalization.is_trnsf=True \
       plot_generalization.row=data \
       plot_aux_trnsf.x=dropout \
       plot_aux_trnsf.col=encoder \
       plot_aux_trnsf.row=data \
       plot_histories.col=dropout \
       plot_histories.style=encoder \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]
