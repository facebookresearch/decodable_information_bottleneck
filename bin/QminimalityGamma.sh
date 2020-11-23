####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Sanity check that gamma does as expected. I.e. there should be a gamma such that training is perfect but testing is bad

experiment="QminimalityGamma"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpxl
datasize=all 
dataset.kwargs.is_augment=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-5
train.optim=adam
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.Q_zy.hidden_size=1024
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=1000
clfs=default
datasize.max_epochs=50
$dev
"

kwargs_multi="
run=0,1,2
clfs.gamma_force_generalization=-1,-0.1,-0.01,0,1
model=dib,vib,stochasticErm
dataset=cifar100,cifar10mnist
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

params="col_val_subset.data=[cifar10mnist,cifar100]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[decay]
col_val_subset.optim=[adam]
col_val_subset.lr=[5e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[dib,vib,stochasticErm]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[mlpxl]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0,1]
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
       plot_aux_trnsf.x=gamma \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.xticks=[-1,-0.1,-0.01,0,1]\
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.row=data \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_generalization.x=gamma \
       plot_generalization.style=model \
       plot_generalization.xticks=[-1,-0.1,-0.01,0,1]\
       plot_generalization.logbase_x=10 \
       plot_generalization.row=data \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.is_trnsf=True \
       plot_histories.col=gamma \
       plot_histories.style=model \
       plot_histories.row=data \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_generalization,plot_aux_trnsf,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[1024] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=gamma \
       plot_metrics.style=model \
       plot_metrics.xticks=[-1,-0.1,-0.01,0,1]\
       plot_metrics.logbase_x=10 \
       plot_metrics.cols_vary_only=["run","minimax"] \
       plot_metrics.row=data \
       plot_generalization.x=gamma \
       plot_generalization.style=model \
       plot_generalization.xticks=[-1,-0.1,-0.01,0,1]\
       plot_generalization.logbase_x=10 \
       plot_generalization.cols_vary_only=["run","minimax"] \
       plot_generalization.is_trnsf=False \
       plot_generalization.row=data \
       plot_histories.col=gamma \
       plot_histories.style=model \
       plot_histories.row=data \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]
