####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether dropout can be seen as decreasing the decodable mutual information. Note that no dropout on zx heads
# Hypothesis
#   - The generalization gap will get smaller with larger dropout, and the DI[Z->X] will get smaller.

experiment="corrWidthProbe"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_normalize=False
dataset.kwargs.is_augment=False
model.architecture.z_dim=512
model.architecture.zy_kwargs.k_prune=0 
datasize=all
clfs=default
model.norm_layer=identity
model.n_skip=1
train.kwargs.lr=1e-5
train.optim=adam
train.scheduling_mode=null
dataset.kwargs.is_random_targets=False 
train.monitor_best=tloss
model.architecture.zy_kwargs.hidden_size=64
model.architecture.n_heads=null
model.architecture.zy_kwargs.n_hidden_layers=2
model.architecture.zx_kwargs.dropout=0
model.loss.beta=0
train.trnsf_kwargs.max_epochs=100
hydra.launcher.time=2000
model=correlation
is_skip_clf_if_precomputed=False
train.kwargs.is_continue_best=True
train.clf_kwargs.is_train=True
model.architecture.is_correlation=False
train.is_corr_eval=True
$dev
"

kwargs_multi="
run=0,1,2,3,4
dataset=cifar10,svhn
encoder=perc32,perc128,perc512,perc8192
"

# copying the folder 
base="tmp_results/corrWidthHq"
results="tmp_results/$experiment"

if [[ ! -d "$results" ]]; then

  echo "Folder does not exist, copying $base to $results"
  cp -r $base $results

fi  


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # precompute the transformer if not already done
    python main.py $kwargs $kwargs_multi $kwargs1 -m &
      
  done
fi

wait 

params="col_val_subset.data=[cifar10,svhn]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[null]
col_val_subset.optim=[adam]
col_val_subset.lr=[1e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[correlation]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[perc32,perc128,perc512,perc8192]
col_val_subset.nskip=[1]
col_val_subset.nheads=[null]
col_val_subset.zdim=[512]
"

params=$params" col_val_subset.enc_zy_nhid=[64] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[2]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=encoder \
       plot_generalization.is_trnsf=True \
       plot_aux_trnsf.x=encoder \
       plot_histories.style=encoder \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_generalization.x=encoder \
       plot_generalization.is_trnsf=False \
       plot_metrics.x=encoder \
       recolt_data.pattern_histories=null \
       mode=[save_tables,plot_metrics,plot_generalization]


# CORRELATION PLOTS
python aggregate.py \
       experiment=$experiment \
       save_experiment="$experiment"/trnsf \
       is_recolt=False \
       correlation_experiment.cause=encoder \
       correlation_experiment.logbase_x=4 \
       mode=[correlation_experiment]


