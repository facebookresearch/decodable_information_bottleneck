####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Graphically show what Q minimality entails

experiment="QminimalityJoint2D"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_normalize=False
dataset.kwargs.is_augment=False
model.architecture.z_dim=2 
train.clf_kwargs.clean_after_run=training
train.kwargs.lr=5e-5
datasize=all
train.monitor_best=tloss
train.trnsf_kwargs.max_epochs=300
train.clf_kwargs.max_epochs=300
hydra.launcher.time=3000
model=cdib
encoder=resnet18
dataset=bincifar100
$dev
model.is_joint=True
"

kwargs_multi="
run=0,1,2
model.Q_zy.hidden_size=1,4,16,128
model.Q_zx.hidden_size=1,4,16,128
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in "" 
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

    wait 

    python main.py $kwargs $kwargs_multi $kwargs1 -m &
    
    sleep 2m # make sure different hydra directory
      
  done
fi

wait 

for enc in "resnet18" 
do
for run in "0" "1" "2" 
do
for data in "bincifar100" 
do
for Qzy in "1" "4" "16" "128"  
do
  prfx="data$data"_"run$run"_"enc$enc"_"Qzy$Qzy"_
  echo "prfx:$prfx"

  # large mesh, decrease for speed (e.g. 20)
  python load_models.py $kwargs \
        load_models.recolt_data.encoders_param=model.Q_zx.hidden_size \
        load_models.recolt_data.encoders_vals=[1,4,16,128] \
        load_models.recolt_data.clf_patterns=["clf_nhid_$Qzy"] \
        dataset=$data \
        encoder=$enc \
        model.Q_zy.hidden_size=$Qzy \
        run=$run \
        additional=load_models \
        load_models.kwargs.prfx=$prfx \
        load_models.plot_reps_clfs.n_mesh=50 \
        load_models.plot_reps_clfs.n_max_scatter=500 \
        load_models.plot_reps_clfs.get_title="loglike" \
        load_models.plot_reps_clfs.is_invert_yaxis=True 
done
done
done
done


params="col_val_subset.data=[bincifar100]
col_val_subset.chckpnt=[tloss]
col_val_subset.lr=[5e-5]
col_val_subset.model=[cdib]
col_val_subset.encoder=[resnet18]
col_val_subset.zdim=[2]
col_val_subset.enc_zy_nhid=[1,4,16,128] 
col_val_subset.enc_zx_nhid=[1,4,16,128] 
"


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=enc_zx_nhid \
       plot_aux_trnsf.logbase_x=2 \
       plot_aux_trnsf.col=enc_zy_nhid \
       plot_aux_trnsf.xticks=[1,4,16,128] \
       plot_generalization.is_trnsf=True \
       plot_generalization.x=enc_zx_nhid \
       plot_generalization.logbase_x=2 \
       plot_generalization.col=enc_zy_nhid \
       plot_generalization.xticks=[1,4,16,128] \
       plot_histories.col=enc_zy_nhid \
       plot_histories.row=enc_zx_nhid \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params=$params" col_val_subset.clf_nhid=[1,4,16,128] 
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_metrics.x=enc_zx_nhid \
       plot_metrics.logbase_x=2 \
       plot_metrics.col=enc_zy_nhid \
       plot_metrics.xticks=[1,4,16,128] \
       plot_generalization.is_trnsf=False \
       plot_generalization.x=enc_zx_nhid \
       plot_generalization.logbase_x=2 \
       plot_generalization.col=enc_zy_nhid \
       plot_generalization.xticks=[1,4,16,128] \
       plot_histories.col=enc_zy_nhid \
       plot_histories.row=enc_zx_nhid \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories] 


