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

experiment="QminJointUnderstandVerylong"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlp
dataset=cifar10mnist 
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-5
train.optim=adam
is_random_labels_clf=False
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.Q_zy.hidden_size=8192
model.Q_zy.n_hidden_layers=2
model.architecture.z_dim=1024
hydra.launcher.time=500
clfs=default
datasize.max_epochs=200
model.loss.altern_minimax=0
model.loss.n_per_head=1
datasize.batch_size=1024
is_skip_trnsf_if_precomputed=False
is_skip_clf_if_precomputed=False
$dev
"

kwargs_multi="
model.is_joint=True
run=0,1,2,3,4
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    #python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 model=stochasticErm -m &

    #wait 


    #cp -r tmp_results/$experiment/data_cifar10mnist/datasize_all/augment_False/rand_False/schedule_decay/optim_adam/lr_5e-05/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/$experiment/data_cifar10mnist/datasize_all/augment_False/rand_False/schedule_decay/optim_adam/lr_5e-05/chckpnt_tloss/wdecay_0/model_gamma-1
    #cp -r tmp_results/$experiment/data_cifar10mnist/datasize_all/augment_False/rand_False/schedule_decay/optim_adam/lr_5e-05/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/$experiment/data_cifar10mnist/datasize_all/augment_False/rand_False/schedule_decay/optim_adam/lr_5e-05/chckpnt_tloss/wdecay_0/model_gamma-01
    #cp -r tmp_results/$experiment/data_cifar10mnist/datasize_all/augment_False/rand_False/schedule_decay/optim_adam/lr_5e-05/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/$experiment/data_cifar10mnist/datasize_all/augment_False/rand_False/schedule_decay/optim_adam/lr_5e-05/chckpnt_tloss/wdecay_0/model_gamma-001

    #wait

    python main.py $kwargs $kwargs_multi $kwargs1 model="gamma-1","gamma-01","gamma-001","stochasticErm" -m &
      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi

wait 

params=""

params=$params" col_val_subset.enc_zy_nhid=[8192] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[2]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_generalization.x=model \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.x_rotate=45 \
       plot_generalization.is_trnsf=True \
       plot_histories.style=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories,plot_generalization]

params_clf=$params" col_val_subset.clf_nhid=[8192] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[2]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=model \
       plot_generalization.x=model \
       plot_generalization.x_rotate=45 \
       plot_generalization.is_trnsf=False \
       plot_histories.style=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_histories,plot_generalization]
