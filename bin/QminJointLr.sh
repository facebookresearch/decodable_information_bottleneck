####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
# - See the impact of not reinitializing the classifier

experiment="QminJointLr"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar10mnistnodist 
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.optim=adam
is_random_labels_clf=False
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.architecture.z_dim=1024
model.Q_zy.hidden_size=128 
model.Q_zy.n_hidden_layers=1
hydra.launcher.time=2000
clfs=default
clfs.anti_generalization_rate=1
model.anti_generalization_rate=-1
clfs.is_reinitialize=True
train.kwargs.is_continue_best=True
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0,0.1,1,10
model=dib
encoder=resnet18
"



if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  "train.kwargs.lr=1e-5 train.scheduling_mode=null" "train.kwargs.lr=1e-4 train.scheduling_mode=decay" 
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

params="col_val_subset.data=[cifar10mnistnodist]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[null,decay]
col_val_subset.optim=[adam]
col_val_subset.lr=[1e-5,1e-4]
col_val_subset.wdecay=[0]
col_val_subset.model=[dib]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[resnet18]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.minimax=[3]
col_val_subset.mchead=[5]
col_val_subset.beta=[0.1,1,10]
"

params=$params" col_val_subset.enc_zy_nhid=[128] 
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
       plot_generalization.col=schedule \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.col=schedule \
       plot_histories.col=beta \
       plot_histories.row=schedule \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[128] 
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
       plot_generalization.col=schedule \
       plot_metrics.x=beta \
       plot_metrics.logbase_x=10 \
       plot_metrics.col=schedule \
       plot_histories.row=schedule \
       plot_histories.col=beta \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]