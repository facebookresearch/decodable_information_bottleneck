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

experiment="QminimalityConditionalJoint"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar10  
dataset.kwargs.is_augment=False 
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=1000
clfs.gamma_force_generalization=-0.1
datasize.max_epochs=200
encoder=mlpl
model.loss.altern_minimax=0
model.loss.n_per_head=1
train.kwargs.lr=5e-5
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0.001,0.1,1,10,100,1000,10000
model=dib,cdib,cdibexact
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

params="col_val_subset.data=[cifar10]
col_val_subset.lr=[5e-5]
col_val_subset.model=[dib,cdib,cdibexact]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
col_val_subset.minimax=[0]
col_val_subset.mchead=[1]
col_val_subset.beta=[0.001,0.1,1,10,100,1000,10000]
"
#cdibapprox

params=$params" col_val_subset.enc_zy_nhid=[128] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_generalization.x=beta \
       plot_generalization.style=model \
       plot_generalization.is_trnsf=True \
       plot_generalization.logbase_x=10 \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.row=model \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_generalization,plot_aux_trnsf,plot_histories]


python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_superpose.x=beta \
       plot_superpose.is_trnsf=True \
       plot_superpose.logbase_x=10 \
       plot_superpose.to_superpose.train_DIQ_yz=DIQ_yz \
       plot_superpose.to_superpose.train_DIQ_xz=DIQ_xz \
       plot_superpose.to_superpose.train_DIQ_xCzy=DIQ_xCzy \
       plot_superpose.to_superpose.train_d_DIQ_xCz=d_DIQ_xCz\
       plot_superpose.cols_vary_only=["run","model"] \
       plot_superpose.value_name="Q Family Bits" \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_superpose]

params_clf=$params" col_val_subset.clf_nhid=[128] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.style=model \
       plot_generalization.is_trnsf=False \
       plot_generalization.logbase_x=10 \
       plot_metrics.x=beta \
       plot_metrics.style=model \
       plot_metrics.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.row=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_generalization,plot_metrics,plot_histories]