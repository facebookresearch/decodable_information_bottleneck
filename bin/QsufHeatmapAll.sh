####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
#   - same as QsufHeatmap but for different z dim | data | encoder

experiment="QsufHeatmapAll"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_normalize=False
dataset.kwargs.is_augment=False
model.Q_zy.k_prune=0 
model.Q_zy.n_hidden_layers=1
clfs=QsufHeatmap
model.norm_layer=identity
model.n_skip=0
train.kwargs.lr=1e-5
train.optim=adam
train.scheduling_mode=null
datasize=all
dataset.kwargs.is_random_targets=False 
clfs.is_reinitialize=False
train.monitor_best=tloss
train.trnsf_kwargs.max_epochs=300
train.clf_kwargs.max_epochs=300
hydra.launcher.time=4000
model=Qsufficiency
$dev
"

kwargs_multi="
run=0,1,2
model.Q_zy.hidden_size=1,2,4,8,16
model.architecture.z_dim=2,8
encoder=resnet18,mlp
dataset=cifar100,svhn
"

if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

    wait 

    python main.py $kwargs $kwargs_multi $kwargs1 -m &
    
    sleep 2m # make sure different hydra directory
      
  done
fi

wait 


params="col_val_subset.data=[cifar100,svhn]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[null]
col_val_subset.optim=[adam]
col_val_subset.lr=[1e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[Qsufficiency]
col_val_subset.dropout=[0]
col_val_subset.encoder=[resnet18,mlp]
col_val_subset.nskip=[0]
col_val_subset.nheads=[0]
col_val_subset.zdim=[2,8]
col_val_subset.minimax=[0]
col_val_subset.mchead=[5]
"


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.col=zdim \
       plot_aux_trnsf.x=enc_zy_nhid \
       plot_aux_trnsf.logbase_x=2 \
       plot_aux_trnsf.xticks=[1,2,4,8,16] \
       plot_aux_trnsf.xticklabels=[1,2,4,8,16] \
       plot_aux_trnsf.style=encoder \
       plot_histories.style=zdim \
       plot_histories.col=enc_zy_nhid \
       plot_histories.folder_col=encoder \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]

params=$params" col_val_subset.enc_zy_nhid=[1,2,4,8,16] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

params_clf=$params" col_val_subset.clf_nhid=[1,2,4,8,16] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER metrics
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=clf_nhid \
       plot_metrics.y=enc_zy_nhid \
       plot_metrics.is_lines=False \
       plot_metrics.col=zdim \
       plot_metrics.is_percentage=True \
       plot_metrics.normalize=col \
       plot_metrics.cmap=YlGnBu_r \
       plot_metrics.folder_col=encoder \
       col_val_subset.epochs=[best] \
       recolt_data.pattern_histories=null \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics] 

# CLASSIFIER histories
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_histories.col=enc_zy_nhid \
       plot_histories.style=clf_nhid \
       plot_histories.hue=zdim \
       plot_histories.folder_col=encoder \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_histories] 


