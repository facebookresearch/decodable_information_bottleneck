####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether gap increase due to Z dim can be explained as increasing H_Q[Y|Z]
# Hypothesis
#   - H_Q[Y|Z] is proportional to the gap increase due to increasing Z

experiment="corrZdimHq"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_normalize=False
dataset.kwargs.is_augment=False
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
model.loss.beta=0
train.trnsf_kwargs.max_epochs=100
hydra.launcher.time=2000
model=correlation
model.architecture.zy_kwargs.n_hidden_layers=2
is_skip_if_precomputed=False
train.kwargs.is_continue_best=True
$dev
"

kwargs_multi="
run=0,1,2,3,4
dataset=cifar10,svhn
encoder=mlp,resnet18
model.architecture.z_dim=32,128,512,2048
"


# copying the folder 
base="tmp_results/corrZdim"
results="tmp_results/$experiment"

if [[ ! -d "$results" ]]; then

  echo "Folder does not exist, copying $base to $results"
  cp -r $base $results

fi  


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # compute the heads but freezing the transformer
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m 
      
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
col_val_subset.encoder=[mlp,resnet18]
col_val_subset.nskip=[1]
col_val_subset.nheads=[null]
col_val_subset.zdim=[32,128,512,2048]
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
       plot_generalization.col=encoder \
       plot_generalization.x=zdim \
       plot_generalization.is_trnsf=True \
       plot_generalization.is_logscale_x=True \
       plot_aux_trnsf.x=zdim \
       plot_aux_trnsf.col=encoder \
       plot_aux_trnsf.is_logscale_x=True \
       plot_histories.col=zdim \
       plot_histories.style=encoder \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_generalization.col=encoder \
       plot_generalization.x=zdim \
       plot_generalization.is_trnsf=False \
       plot_generalization.is_logscale_x=True \
       plot_metrics.x=zdim \
       plot_metrics.col=encoder \
       plot_metrics.is_logscale_x=True \
       recolt_data.pattern_histories=null \
       mode=[save_tables,plot_metrics,plot_generalization]

# CORRELATION PLOTS
python aggregate.py \
       experiment=$experiment \
       save_experiment="$experiment"/trnsf \
       is_recolt=False \
       correlation_experiment.cause=zdim \
       correlation_experiment.col_sep_plots=encoder \
       mode=[correlation_experiment]