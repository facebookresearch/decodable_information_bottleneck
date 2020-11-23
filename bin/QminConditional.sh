####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
# - Show the gains of using conditional DIB instead of DIB
# - compare all possible conditional versions "H_Q[X|Z,Y]", "H_Q[X|Z]-H_Q[Y|Z]", "H_Q'[X|Z,Y]"
# Hypothesis
#   - when increasing beta generalization gap will decrease for worst case until 1 then plateau(both CDIB and DIB ~equivalently)
#   - when increasing beta test performance will go up until 1 then ~plateau for CDIB and decrease for DIB 
#   - when increasing beta H_Q[Y|Z] should decrease until beta=1 and then stabilize for CDIB (at 0) but continue decreasing for DIB (because removing information aboutZ)
#   - when increasing beta H_Q[X|Z] should be flat until 1 for CDIB (and then decrease) but decrease all the time for DIB (because removing information aboutZ )

experiment="QminConditional"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
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
clfs.is_reinitialize=True
train.kwargs.is_continue_best=True
datasize.max_epochs=100
encoder=resnet18
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=1,10,100,1000,10000
model=dib,cdib,cdibexact,cdibapprox
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
col_val_subset.model=[dib,cdib,cdibexact,cdibapprox]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[resnet18]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.minimax=[3]
col_val_subset.mchead=[3]
col_val_subset.beta=[1,10,100,1000,10000]
"
#cdibapprox

params=$params" col_val_subset.enc_zy_nhid=[256] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=beta \
       plot_generalization.col=model \
       plot_generalization.is_trnsf=True \
       plot_generalization.logbase_x=10 \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.row=model \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]


python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       col_val_subset.model=[dib,cdib,cdibexact] \
       plot_superpose.x=beta \
       plot_superpose.is_trnsf=True \
       plot_superpose.logbase_x=10 \
       plot_superpose.to_superpose.train_H_Q_xCz=H_Q_xCz \
       plot_superpose.to_superpose.train_H_Q_xCzy=H_Q_xCzy \
       plot_superpose.to_superpose.train_d_H_Q_xCz=d_H_Q_xCz\
       plot_superpose.to_superpose.train_d_H_Q_xCz=d_H_Q_xCz\
       plot_superpose.cols_vary_only=["run","model"] \
       plot_superpose.value_name="Q Family Bits" \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_superpose]

params_clf=$params" col_val_subset.clf_nhid=[256] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.col=model \
       plot_generalization.is_trnsf=False \
       plot_generalization.logbase_x=10 \
       plot_metrics.x=beta \
       plot_metrics.style=model \
       plot_metrics.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.row=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]