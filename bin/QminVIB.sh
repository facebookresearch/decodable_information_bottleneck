####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Compare effect of VIB and DIB without joint on generalization
# Hypothesis
#   - DIB better

experiment="QminVIB"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=resnet18
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
hydra.launcher.time=2000
clfs=default
clfs.gamma_force_generalization=-1
datasize.max_epochs=100
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0,0.1,1,10,100,1000
model=dib,vib
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
col_val_subset.model=[dib,vib]
col_val_subset.dropout=[0.]
col_val_subset.encoder=[resnet18]
col_val_subset.nskip=[0]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0,0.1,1,10,100,1000]
"

params=$params" col_val_subset.enc_zy_nhid=[256] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.xticks=[0,0.1,1,10,100,1000]\
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.cols_vary_only=["run","minimax"] \
       plot_superpose.x=beta \
       plot_superpose.is_trnsf=True \
       plot_superpose.to_superpose.train_DIQ_xz=Diq_xz \
       plot_superpose.to_superpose.train_I_xz=I_xz \
       plot_superpose.value_name=star_bits \
       plot_superpose.xticks=[0,0.1,1,10,100,1000]\
       plot_superpose.cols_vary_only=["run","minimax"] \
       plot_superpose.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.style=model \
       plot_histories.cols_vary_only=["run","minimax"] \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_superpose,plot_aux_trnsf,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[256] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       plot_metrics.x=beta \
       plot_metrics.style=model \
       plot_metrics.xticks=[0,0.1,1,10,100,1000]\
       plot_metrics.logbase_x=10 \
       plot_histories.col=beta \
       plot_histories.style=model \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_histories]
