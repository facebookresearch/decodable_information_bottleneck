####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Compare effect of VIB and DIB without joint on generalization
# Hypothesis
#   - DIB better

experiment="QminJointVIBUnderstandLarge"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpxl
dataset=cifar10mnist 
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=1e-4
train.optim=adam
is_random_labels_clf=False
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.Q_zy.hidden_size=512
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=300
clfs=default
datasize.max_epochs=50
model.loss.n_per_head=1
datasize.batch_size=256
is_skip_trnsf_if_precomputed=False
is_skip_clf_if_precomputed=False
clfs.gamma_force_generalization=-0.1
model.loss.altern_minimax=3
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0,0.1,1,10,100
model=cdib,vib
model.is_joint=True
"

#is_test_on_valid=True
#hydra.launcher.partition=dev

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

params=""

params=$params" col_val_subset.enc_zy_nhid=[512] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_generalization.x=beta \
       plot_generalization.style=model \
       plot_generalization.logbase_x=10 \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.is_trnsf=True \
       plot_histories.col=beta \
       plot_histories.style=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[512] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=beta \
       plot_metrics.style=model \
       plot_metrics.logbase_x=10 \
       plot_metrics.cols_vary_only=["run","minimax"] \
       plot_generalization.x=beta \
       plot_generalization.style=model \
       plot_generalization.logbase_x=10 \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.is_trnsf=False \
       plot_histories.col=beta \
       plot_histories.style=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]
