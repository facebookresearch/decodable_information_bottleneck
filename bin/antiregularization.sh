####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Sanity check that gamma does as expected. I.e. there should be a gamma such that training is perfect but testing is bad

experiment="antiregularization"

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
hydra.launcher.time=500
datasize.max_epochs=200
encoder=mlpl
train.kwargs.lr=5e-5
model=erm
$dev
"

kwargs_multi="
run=0,1,2
"



if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

    wait 

    python main.py $kwargs $kwargs_multi $kwargs1 clfs.gamma_force_generalization=-10,-1,-0.5,-0.1,-0.01,0,1 -m &
      
    # make sure different hydra directory
    sleep 2m 
    
  done
fi

wait 

params="col_val_subset.data=[cifar10]
col_val_subset.lr=[5e-5]
col_val_subset.model=[erm]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
"

params_clf=$params" col_val_subset.clf_nhid=[128] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=gamma \
       plot_metrics.logbase_x=10 \
       plot_generalization.x=gamma \
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=False \
       plot_histories.col=gamma \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories]
