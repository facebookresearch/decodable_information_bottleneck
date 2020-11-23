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

experiment="corrResnetsDev"

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
hydra.launcher.time=1500
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
dataset=svhn,cifar10
encoder=resnet18,resnet34,resnet50,resnet101
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
       col_val_subset.gamma=[-0.1] \
       plot_generalization.x=encoder \
       plot_generalization.is_trnsf=False \
       plot_generalization.row=data \
       plot_metrics.x=encoder \
       plot_metrics.row=data \
       recolt_data.pattern_histories=null \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_generalization,plot_metrics]


# CORRELATION PLOTS
python aggregate.py \
       experiment=$experiment \
       save_experiment="$experiment"/trnsf \
       is_recolt=False \
       correlation_experiment.cause=encoder \
       mode=[correlation_experiment]
