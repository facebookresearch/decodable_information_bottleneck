####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
# - Select the datset for minimality experiments. Test when bob not using joint.

experiment="QminimalityDatasetUnderstand"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-5
train.optim=adam
is_random_labels_clf=False
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.architecture.z_dim=1024
hydra.launcher.time=500
clfs=default
clfs.gamma_force_generalization=-0.1
model.Q_zy.hidden_size=1024
model.Q_zy.n_hidden_layers=1
encoder=mlpxl
datasize.max_epochs=50
model.loss.n_per_head=1
$dev
"

kwargs_multi="
run=0,1,2
model=cdib,cdibsmall
model.loss.beta=0,0.001,0.01,0.1,1,10
dataset=cifar100,cifar10
"

kwargs_multi="
run=0,1
model=cdibsmall
model.loss.beta=0.01,1,10
dataset=cifar10
datasize.max_epochs=10
train.trnsf_kwargs.max_epochs=10
train.clf_kwargs.max_epochs=10
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 train.kwargs.lr="1e-4|100","1e-4|1","1e-4|10" -m &
    
  done
fi

wait 

params="col_val_subset.data=[cifar100,cifar10]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[decay]
col_val_subset.optim=[adam]
col_val_subset.lr=[5e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[cdib,cdibsmall]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[mlpxl]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0,0.001,0.01,0.1,1,10]
"

params=$params" col_val_subset.lr=['1e-4|0.1','1e-4|1','1e-4|10','1e-4|100']
"

params=$params" col_val_subset.enc_zy_nhid=[1024] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=True \
       plot_generalization.logbase_x=10 \
       plot_generalization.row=lr \
       plot_generalization.col=model \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.row=lr \
       plot_aux_trnsf.style=model \
       plot_histories.row=lr \
       plot_histories.style=beta \
       plot_histories.col=model \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[1024] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=False \
       plot_generalization.logbase_x=10 \
       plot_generalization.row=data \
       plot_generalization.col=model \
       plot_metrics.x=beta \
       plot_metrics.logbase_x=10 \
       plot_metrics.row=data \
       plot_metrics.style=model \
       plot_histories.row=data \
       plot_histories.style=beta \
       plot_histories.col=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]