####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Understand what we are proposing with Q minimality by runnign Q minimality on test set

experiment="Qminimality2DSmallHope"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_normalize=True
dataset.kwargs.is_augment=False
model.architecture.z_dim=2 
train.clf_kwargs.clean_after_run=training
train.kwargs.lr=1e-5
train.monitor_best=last
hydra.launcher.time=3000
dataset.valid=test
train.trnsf_kwargs.max_epochs=500
train.clf_kwargs.max_epochs=1000
datasize.n_examples_test=train
train.scheduling_mode=decay
clfs.gamma_force_generalization=-0.3
datasize.batch_size=16
model.Q_zy.n_hidden_layers=0
encoder=mlp
$dev
"

kwargs_multi="
run=0,1,2,3,4
model=simplecdib,simplecdibmm,simplecdibexact,stochasticErm,vib
dataset=bincifar100,binmnist
datasize=mini,small
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







# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.row=data \
       plot_generalization.is_trnsf=True \
       plot_generalization.x=model \
       plot_generalization.row=data \
       plot_histories.col=model \
       plot_histories.row=data \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]



# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_metrics.x=model \
       plot_metrics.row=data \
       plot_generalization.is_trnsf=False \
       plot_generalization.x=model \
       plot_generalization.row=data \
       plot_histories.col=model \
       plot_histories.row=data \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories] 




wait



for run in "0" "1" "2" "3" "4" 
do
for data in "bincifar100" "binmnist"
do
for datasize in "mini" "small"
do
for model in  "simplecdibmm" #"simplecdib" "simplecdibexact" "stochasticErm" "vib"
do
  prfx="data$data"_"run$run"_"datasize$datasize"_"model$model"_
  echo "prfx:$prfx"

  # large mesh, decrease for speed (e.g. 20)
  python load_models.py $kwargs \
        load_models.recolt_data.encoders_param=model.Q_zx.n_hidden_layers \
        load_models.recolt_data.encoders_vals=[0] \
        load_models.recolt_data.clf_patterns=["clf"] \
        dataset=$data \
        datasize=$datasize \
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

