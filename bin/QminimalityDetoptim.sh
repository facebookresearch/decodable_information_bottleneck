####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Check how well works for full batch optimization 

experiment="QminimalityDetoptim"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpxl
dataset=cifar100
datasize=all 
dataset.kwargs.is_augment=False 
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.Q_zy.hidden_size=1024
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=2000
clfs=default
clfs.gamma_force_generalization=-0.1
datasize.max_epochs=50
model.loss.altern_minimax=0
model.loss.n_per_head=1
model=stochasticErm
train.scheduling_mode=null
is_nvidia_smi=True
datasize.batch_size=8192
$dev
"


kwargs_multi="
run=0,1,2
model.loss.beta=0,0.01,0.1,1,10,100
"

kwargs_multi="
run=0
model.loss.beta=0.01,0.1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  "" #"train.optim=adam train.kwargs.lr=0.1"
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 train.optim=LBFGSnew,LBFGSfull train.kwargs.lr=.1 train.lr_factor_zx=1 -m &

    #sleep 2m 

    #python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 train.optim=adam train.kwargs.lr=0.001 -m &

    wait 

    python main.py $kwargs $kwargs_multi $kwargs1 train.optim=LBFGSnew,LBFGSfull train.kwargs.lr=.1 train.lr_factor_zx=1 -m &

    #sleep 2m 

    #python main.py $kwargs $kwargs_multi $kwargs1 train.optim=adam train.kwargs.lr=0.001 -m &
      
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
col_val_subset.schedule=[null]
col_val_subset.optim=[LBFGS,LBFGSnew,LBFGSfull,adam]
col_val_subset.lr=[0.1,1]
col_val_subset.wdecay=[0]
col_val_subset.model=[cdib]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[mlpxl]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0,0.01,0.1,1,10,100]
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
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=optim \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_generalization.x=beta \
       plot_generalization.col=optim \
       plot_generalization.logbase_x=10 \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.is_trnsf=True \
       plot_histories.col=beta \
       plot_histories.style=optim \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_superpose,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[1024] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.col=optim \
       plot_generalization.logbase_x=10 \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.is_trnsf=False \
       plot_metrics.x=beta \
       plot_metrics.logbase_x=10 \
       plot_metrics.style=optim \
       plot_histories.col=beta \
       plot_histories.style=optim \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]
