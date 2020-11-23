####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether base B exapansion is helpful compared to random labeling
# Hypothesis
#   - little beneficial but not much

experiment="QminRandom"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=resnet18
dataset=cifar10mnistnodist 
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=decay
train.kwargs.lr=5e-5
train.optim=adam
is_random_labels_clf=False
train.monitor_best=tloss 
model.Q_zy.k_prune=0 
model.Q_zy.hidden_size=256
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=3000
clfs=default
clfs.gamma_force_generalization=-1
model.loss.beta=1
train.kwargs.is_continue_best=True
datasize.max_epochs=100
$dev
"

kwargs_multi="
run=0,1,2
model.loss.n_per_head=1,3,5,10
model=dib,dibrand
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

params="col_val_subset.data=[cifar10mnistnodist]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[decay]
col_val_subset.optim=[adam]
col_val_subset.lr=[5e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[dib,dibrand]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[resnet18]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.minimax=[3]
col_val_subset.mchead=[1,3,5,10]
col_val_subset.beta=[1]
"

params=$params" col_val_subset.enc_zy_nhid=[256] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       kwargs.pretty_renamer.Mchead="N. Possible Labels" \
       plot_aux_trnsf.x=mchead \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.xticks=[1,3,5,10]\
       plot_aux_trnsf.xticklabels=[5,15,25,50]\
       plot_histories.col=mchead \
       plot_histories.style=model \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[256] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=mchead \
       plot_metrics.style=model \
       plot_metrics.xticks=[1,3,5,10]\
       plot_metrics.xticklabels=[5,15,25,50]\
       plot_histories.col=mchead \
       plot_histories.style=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_histories]
