####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : table of worst case when using regularizers on Bon

experiment="coloredmnist"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed
encoder=mlpl
dataset.kwargs.is_augment=False 
train.kwargs.lr=5e-5
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
hydra.launcher.time=200
clfs.gamma_force_generalization=-0.0
datasize.max_epochs=200
model.architecture.z_dim=1024
model.architecture.is_wrap_batchnorm=True
$dev
"

kwargs_multi="
run=0,1,2
model=erm,cdibhigher,simplecdibdet
dataset=coloredmnist,mnist
"



if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &
      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi

wait 



params="col_val_subset.data=[mnist,coloredmnist]
col_val_subset.lr=[5e-5]
col_val_subset.model=[erm,cdibhigher,simplecdibdet]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
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
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.row=data \
       plot_generalization.x=model \
       plot_generalization.row=data \
       plot_generalization.is_trnsf=True \
       plot_histories.col=model \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

