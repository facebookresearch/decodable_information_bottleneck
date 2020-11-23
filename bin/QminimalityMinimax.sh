####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Show the effect of the number of minimax step when using higher order grad

experiment="QminimalityMinimax"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpl
dataset=cifar10 
dataset.kwargs.is_augment=False 
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=2000
clfs.gamma_force_generalization=-0.1
datasize.max_epochs=100
model.loss.n_per_head=1
is_nvidia_smi=True
$dev
"


kwargs_multi="
run=0,1,2
model.loss.beta=100000
model.loss.altern_minimax=0,5,10
model=cdib,cdibhigher
"
#model.loss.beta=0.001,0.01,0.1,1,10,100,1000,10000
#model.loss.altern_minimax=0.01,0.1,1,10,100,

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
col_val_subset.model=[cdib,cdibhigher]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
col_val_subset.minimax=[0,5,10,30]
col_val_subset.mchead=[1]
col_val_subset.beta=[0.001,0.01,0.1,1,10,100,1000,10000,100000]
"

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
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=True \
       plot_generalization.style=minimax \
       plot_generalization.col=model \
       plot_generalization.sharey=True \
       plot_generalization.is_legend_out=True \
       plot_generalization.is_no_legend_title=False \
       plot_superpose.x=beta \
       plot_superpose.logbase_x=10 \
       plot_superpose.is_trnsf=True \
       plot_superpose.to_superpose.train_DIQ_yz=DIQ_yz \
       plot_superpose.to_superpose.train_d_DIQ_xz=d_DIQ_xz \
       plot_superpose.value_name="Q Family Bits" \
       plot_superpose.style=minimax \
       plot_superpose.col=model \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.style=minimax \
       plot_aux_trnsf.col=model \
       plot_histories.col=beta \
       plot_histories.row=model \
       plot_histories.style=minimax \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_generalization,plot_superpose,plot_aux_trnsf,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[128] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_generalization.x=beta \
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=False \
       plot_generalization.style=minimax \
       plot_generalization.col=model \
       plot_generalization.sharey=True \
       plot_generalization.is_legend_out=True \
       plot_generalization.is_no_legend_title=False \
       plot_metrics.x=beta \
       plot_metrics.logbase_x=10 \
       plot_metrics.style=minimax \
       plot_metrics.col=model \
       plot_histories.col=beta \
       plot_histories.style=minimax \
       plot_histories.row=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_generalization,plot_metrics,plot_histories]
