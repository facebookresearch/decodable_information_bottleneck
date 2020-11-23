####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
# - Select the best architecture for Alice. Test when bob not using joint.

experiment="QminimalityArchitectureAlice"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar100
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-5
train.optim=adam
train.monitor_best=tloss 
model.architecture.z_dim=1024
hydra.launcher.time=1500
clfs=default
clfs.gamma_force_generalization=-0.1
encoder=mlpxl
datasize.max_epochs=50
model.loss.altern_minimax=0
model.loss.n_per_head=1
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0.1,1,10,100
model.Q_zy.n_hidden_layers=1,2,3
model.Q_zy.hidden_size=16,64,256,1024
model=cdib,vib
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

wait 

params="col_val_subset.data=[cifar100]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[decay]
col_val_subset.optim=[adam]
col_val_subset.lr=[5e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[cdib,vib]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[mlpxl]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0.1,1,10,100]
"

params=$params" col_val_subset.enc_zy_nhid=[256,1024] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1,2]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=True \
       plot_generalization.logbase_x=10 \
       plot_generalization.col=enc_zy_nlay \
       plot_generalization.row=enc_zy_nhid \
       plot_generalization.style=model \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.col=enc_zy_nlay \
       plot_aux_trnsf.row=enc_zy_nhid \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_histories.col=enc_zy_nlay \
       plot_histories.row=enc_zy_nhid \
       plot_histories.style=beta \
       plot_histories.folder_col=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[256,1024] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1,2]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=False \
       plot_generalization.logbase_x=10 \
       plot_generalization.col=enc_zy_nlay \
       plot_generalization.row=enc_zy_nhid \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.style=model \
       plot_metrics.x=beta \
       plot_metrics.style=model \
       plot_metrics.logbase_x=10 \
       plot_metrics.col=enc_zy_nlay \
       plot_metrics.row=enc_zy_nhid \
       plot_metrics.cols_vary_only=["run","minimax"] \
       plot_histories.col=enc_zy_nlay \
       plot_histories.row=enc_zy_nhid \
       plot_histories.style=beta \
       plot_histories.folder_col=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]