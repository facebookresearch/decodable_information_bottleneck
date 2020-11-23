####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether batch size can be seen as decreasing the decodable mutual information in the case of Alice = bob

experiment="corrBatchnewBob"

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
train.trnsf_kwargs.max_epochs=5
train.clf_kwargs.max_epochs=5
hydra.launcher.time=1500
model=erm
is_correlation=True
is_correlation_Bob=True
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
datasize=b8,b32,b128,b512,b2048
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
       plot_generalization.x=datasize \
       plot_generalization.is_trnsf=True \
       plot_generalization.is_merge_data_size=false \
       plot_generalization.row=data \
       plot_aux_trnsf.x=datasize \
       plot_aux_trnsf.col=encoder \
       plot_aux_trnsf.is_merge_data_size=false \
       plot_aux_trnsf.row=data \
       plot_histories.col=datasize \
       plot_histories.style=encoder \
       plot_histories.is_merge_data_size=false \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]


# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_generalization.col=encoder \
       plot_generalization.x=datasize \
       plot_generalization.is_trnsf=False \
       plot_generalization.is_merge_data_size=false \
       plot_generalization.row=data \
       plot_metrics.x=datasize \
       plot_metrics.col=encoder \
       plot_metrics.row=data \
       plot_metrics.is_merge_data_size=false \
       recolt_data.pattern_histories=null \
       mode=[save_tables,plot_metrics,plot_generalization]

# CORRELATION PLOTS
python aggregate.py \
       experiment=$experiment \
       save_experiment="$experiment"/trnsf \
       is_recolt=False \
       correlation_experiment.cause=datasize \
       correlation_experiment.col_sep_plots=encoder \
       mode=[correlation_experiment]
