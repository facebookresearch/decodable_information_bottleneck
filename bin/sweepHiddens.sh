####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Sweep over layers of neural nets to understand how it makes the class more complicated.

experiment="sweepHiddens"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlp 
dataset=cifar100
datasize=all 
dataset.kwargs.is_augment=False 
dataset.kwargs.is_random_targets=False 
train.scheduling_mode=null
train.kwargs.lr=1e-5
train.optim=adam
is_random_labels_clf=True
train.monitor_best=vacc
train.clf_kwargs.monitor_best=tloss 
model.architecture.zy_kwargs.k_prune=0 
model.architecture.zy_kwargs.hidden_size=64
model.architecture.zy_kwargs.n_hidden_layers=0
model=vanilla 
train.trnsf_kwargs.max_epochs=100
train.clf_kwargs.max_epochs=300
hydra.launcher.time=1000
clfs=sweepHiddens
$dev
"

kwargs_multi="
run=0,1,2
model.architecture.z_dim=2,16,1024
"

# copying the folder to not recomput all sweep transformer
base="tmp_results/sweepDflt"
results="tmp_results/$experiment"

if [[ ! -d "$results" ]]; then

  echo "Folder does not exist, copying $base to $results"
  cp -r $base $results

fi 


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
col_val_subset.chckpnt=[vacc]
col_val_subset.schedule=[null]
col_val_subset.optim=[adam]
col_val_subset.lr=[1e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[vanilla]
col_val_subset.dropout=[0]
col_val_subset.encoder=[mlp]
col_val_subset.nskip=[0]
col_val_subset.zdim=[2,16,1024]
col_val_subset.minimax=[0]
col_val_subset.mchead=[5]
"

params=$params" col_val_subset.enc_zy_nhid=[64] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[0]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=zdim \
       plot_aux_trnsf.logbase_x=8 \
       plot_histories.style=zdim \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories] 

params=$params" col_val_subset.clf_nlay=[1] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nhid=[1,4,16,64,256,1024,4096]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_metrics.x=clf_nhid \
       plot_metrics.style=zdim \
       plot_metrics.logbase_x=4 \
       plot_metrics.xticks=[1,4,16,64,256,1024,4096] \
       plot_metrics.xticklabels=[1,4,16,64,256,1024,4096] \
       plot_metrics.x_rotate=30 \
       plot_histories.style=zdim \
       plot_histories.col=clf_nhid \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_histories] 
