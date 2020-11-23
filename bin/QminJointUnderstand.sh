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

experiment="QminJointUnderstand"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlp
dataset=cifar10mnist
datasize=small 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-4
train.optim=adam
is_random_labels_clf=False
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.Q_zy.hidden_size=1024
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=50
clfs=default
datasize.max_epochs=20
model.loss.altern_minimax=0
model.loss.n_per_head=1
datasize.batch_size=256
is_skip_trnsf_if_precomputed=False
is_skip_clf_if_precomputed=False
$dev
"

kwargs_multi="
run=0,1,2,3,4
hydra.launcher.partition=dev
is_test_on_valid=True
dataset.valid_size=0.5
model.is_joint=True
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 model=stochasticErm -m &

    wait 

    #cp -r tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma1
    #cp -r tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-1
    cp -r tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-001
    #cp -r tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-005
    cp -r tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-01
    cp -r tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_cifar10mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-05
    #cp -r tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-03
    #cp -r tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_stochasticErm tmp_results/QminJointUnderstand/data_mnist/datasize_small/augment_False/rand_False/schedule_decay/optim_adam/lr_0.0005/chckpnt_tloss/wdecay_0/model_gamma-05

    #wait

    python main.py $kwargs $kwargs_multi $kwargs1 model="gamma-001,gamma-05,gamma-01,stochasticErm" -m &
      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi

wait 

params="col_val_subset.model=[stochasticErm,gamma-001,gamma-005,gamma-01,gamma-05,gamma-1,gamma-03] "

params=$params" col_val_subset.enc_zy_nhid=[1024] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_generalization.x=model \
       plot_generalization.x_rotate=45 \
       plot_generalization.is_trnsf=True \
       plot_histories.style=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories,plot_generalization]

params_clf=$params" col_val_subset.clf_nhid=[1024] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
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
