####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : table of worst case when using regularizers on Bon

experiment="invCifar10mnistData"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed
encoder=cnn
dataset.kwargs.is_augment=False 
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
hydra.launcher.time=2000
clfs.gamma_force_generalization=-0.0
datasize.max_epochs=100
model.architecture.z_dim=1024
model.architecture.is_wrap_batchnorm=True
train.kwargs.lr=5e-5
$dev
"

kwargs_multi="
run=0,1,2
model=cdibquick
dataset=bincifar10mnistdep5,bincifar10mnistdep3,bincifar10mnistdep7,bincifar10mnistdep8,bincifar10mnistdep9
model.loss.beta=0,.1,1,10,100,1000
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



params="col_val_subset.data=[bincifar10mnistdep5,bincifar10mnistdep3,bincifar10mnistdep7,bincifar10mnistdep8,bincifar10mnistdep9]
col_val_subset.lr=[5e-5]
col_val_subset.model=[cdibquick]
col_val_subset.beta=[0,.1,1,10,100,1000]
col_val_subset.encoder=[cnn]
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
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.row=data \
       plot_aux_trnsf.col=model \
       plot_aux_trnsf.logbase_x=10 \
       plot_generalization.x=beta \
       plot_generalization.row=data \
       plot_generalization.is_trnsf=True \
       plot_generalization.col=model \
       plot_generalization.logbase_x=10 \
       plot_histories.col=model \
       plot_histories.row=beta \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_generalization,plot_aux_trnsf,plot_histories]

