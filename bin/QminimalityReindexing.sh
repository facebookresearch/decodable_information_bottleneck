####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See whether taking an average over different reidixing is beneficial (or whether can use any index before base B)
# Hypothesis
#   - little beneficial but not much

experiment="QminimalityReindexing"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpl
dataset=cifar10
dataset.kwargs.is_augment=False 
train.kwargs.lr=5e-5
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
model.architecture.z_dim=1024
hydra.launcher.time=2000
clfs.gamma_force_generalization=-0.1
datasize.max_epochs=200
model.loss.altern_minimax=5
model.loss.is_higher=True
model.architecture.is_wrap_batchnorm=True
model.loss.n_per_head=1
model.loss.z_norm_reg=0
$dev
"

kwargs_multi="
run=0,1,2
model.loss.n_per_head=1,3,5
model.loss.beta=0.01,0.1,1,10,100
model=cdib,cdibsameidcs
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
col_val_subset.model=[cdib,cdibsameidcs]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
col_val_subset.mchead=[1,3,5]
col_val_subset.beta=[0.01,0.1,1,10,100]
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
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.col=mchead \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.logbase_x=10 \
       plot_generalization.is_legend_out=True \
       plot_generalization.is_no_legend_title=False \
       plot_generalization.sharey=True \
       plot_generalization.x=beta \
       plot_generalization.col=mchead \
       plot_generalization.style=model \
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=True \
       plot_superpose.x=beta \
       plot_superpose.logbase_x=10 \
       plot_superpose.is_trnsf=True \
       plot_superpose.to_superpose.train_DIQ_yz=DIQ_yz \
       plot_superpose.to_superpose.train_d_DIQ_xz=d_DIQ_xz \
       plot_superpose.value_name="Q Family Bits" \
       plot_superpose.col=mchead \
       plot_superpose.style=model \
       plot_superpose.is_legend_out=True \
       plot_superpose.is_no_legend_title=False \
       plot_superpose.sharey=True \
       plot_histories.col=mchead \
       plot_histories.row=beta \
       plot_histories.style=model \
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
       kwargs.pretty_renamer.Cdib=Different_Indexing \
       plot_metrics.x=beta \
       plot_metrics.col=mchead \
       plot_metrics.style=model \
       plot_metrics.logbase_x=10 \
       plot_generalization.x=beta \
       plot_generalization.col=mchead \
       plot_generalization.style=model \
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=False \
       plot_generalization.is_legend_out=True \
       plot_generalization.is_no_legend_title=False \
       plot_generalization.sharey=True \
       plot_histories.col=mchead \
       plot_histories.row=beta \
       plot_histories.style=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_generalization,plot_metrics,plot_histories]
