####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Graphically show what Q minimality entails

experiment="Qminimality2DBest"

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
train.monitor_best=last
hydra.launcher.time=3000
dataset=bincifar100
model.loss.beta=1
model.loss.n_per_head=1
datasize=mini
dataset.valid=test
train.trnsf_kwargs.max_epochs=50
train.clf_kwargs.max_epochs=200
datasize.n_examples_test=train
train.scheduling_mode=null
model.gamma_force_generalization=1
clfs.gamma_force_generalization=1
datasize.batch_size=8
model.architecture.is_stochastic=False
$dev
"

kwargs_multi="
run=0,1
model.Q_zy.hidden_size=128
model.Q_zx.hidden_size=128
model=cdib,cdibexact,stochasticErm
encoder=mlpl,resnet18
hydra.launcher.partition=dev
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



params="col_val_subset.data=[bincifar100]
col_val_subset.chckpnt=[last]
col_val_subset.lr=[5e-5]
col_val_subset.model=[cdib,cdibexact,stochasticErm]
col_val_subset.encoder=[mlpl,resnet18]
col_val_subset.zdim=[2]
col_val_subset.enc_zy_nhid=[128] 
col_val_subset.enc_zx_nhid=[128] 
"


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.col=encoder \
       plot_generalization.is_trnsf=True \
       plot_generalization.x=model \
       plot_generalization.col=encoder \
       plot_histories.col=model \
       plot_histories.row=encoder \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params=$params" col_val_subset.clf_nhid=[1,2,4,8,16,128] 
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_metrics.x=model \
       plot_metrics.col=encoder \
       plot_generalization.is_trnsf=False \
       plot_generalization.x=model \
       plot_generalization.col=encoder \
       plot_histories.col=model \
       plot_histories.row=encoder \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories] 






wait



for enc in "mlpl" "resnet18"
do
for run in "0" "1" 
do
for data in "bincifar100" 
do
for Qzy in "128"  
do
for model in "cdib"   "stochasticErm" "cdibexact" 
do
  prfx="data$data"_"run$run"_"enc$enc"_"Qzy$Qzy"_"model$model"_
  echo "prfx:$prfx"

  # large mesh, decrease for speed (e.g. 20)
  python load_models.py $kwargs \
        load_models.recolt_data.encoders_param=model.Q_zx.hidden_size \
        load_models.recolt_data.encoders_vals=[128] \
        load_models.recolt_data.clf_patterns=["clf"] \
        dataset=$data \
        encoder=$enc \
        model.Q_zy.hidden_size=$Qzy \
        run=$run \
        model=$model \
        additional=load_models \
        load_models.kwargs.prfx=$prfx \
        load_models.plot_reps_clfs.n_mesh=1000 \
        load_models.plot_reps_clfs.n_max_scatter=500 \
        load_models.plot_reps_clfs.get_title="loglike" \
        load_models.plot_reps_clfs.is_invert_yaxis=True \
        load_models.plot_reps_clfs.is_plot_test=True
done
done
done
done
done

