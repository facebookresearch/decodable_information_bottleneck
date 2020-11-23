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

experiment="corrWdecAliceWorst"

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
model=erm
datasize=all
hydra.launcher.time=300
train.clf_kwargs.max_epochs=300
dataset.train=valid
clfs.gamma_force_generalization=-0.1
is_correlation=True
clfs.is_reinitialize=False
is_skip_clf_if_precomputed=False
train.clf_kwargs.is_train=False
hydra.launcher.mem_limit=64
$dev
"

kwargs_multi="
run=0,1,2,3,4
dataset=cifar10,svhn
encoder=mlp,resnet18
train.weight_decay=1e-6,1e-5,1e-4,1e-3,1e-2,0.1
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
       plot_generalization.x=wdecay \
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=False \
       plot_generalization.row=data \
       plot_metrics.x=wdecay \
       plot_metrics.logbase_x=10 \
       plot_metrics.row=data \
       recolt_data.pattern_histories=null \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_generalization,plot_metrics]


# CORRELATION PLOTS
python aggregate.py \
       experiment=$experiment \
       save_experiment="$experiment"/trnsf \
       is_recolt=False \
       correlation_experiment.cause=wdecay \
       correlation_experiment.col_sep_plots=encoder \
       mode=[correlation_experiment]