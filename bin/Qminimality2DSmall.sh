####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Graphically show what Q minimality with respect to different Qs entails when working on small data

experiment="Qminimality2DSmall"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_augment=False
model.architecture.z_dim=2 
train.clf_kwargs.clean_after_run=training
train.kwargs.lr=5e-5
train.trnsf_kwargs.max_epochs=500
train.clf_kwargs.max_epochs=500
hydra.launcher.time=1000
encoder=mlpl
dataset=bincifar100
model.Q_zy.hidden_size=64
clfs.gamma_force_generalization=-0.1
model.loss.altern_minimax=5
model.loss.is_higher=True
model.is_joint=True
dataset.valid=test
datasize.n_examples_test=train
$dev
"

kwargs_multi="
run=0,1,2
model=cdib,cdibS,cdibL,vib,stochasticErm
datasize=small,mini
model.loss.beta=0.1,1
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


for enc in "mlpl" 
do
for run in "0" "1" "2" 
do
for model in "cdib"  "cdibS" "cdibL" "stochasticErm" "vib"
do
for beta in "0.1" "1" 
do
for datasize in "small" "mini" 
do
  prfx="model$model"_"run$run"_"enc$enc"_"beta$beta"_"datasize$datasize"_
  echo "prfx:$prfx"

  # large mesh, decrease for speed (e.g. 20)
  python load_models.py $kwargs \
        load_models.recolt_data.encoders_param=model.loss.beta \
        load_models.recolt_data.encoders_vals=[$beta] \
        load_models.recolt_data.clf_patterns=['clf_nhid_64/'] \
        model=$model \
        encoder=$enc \
        datasize=$datasize\
        run=$run \
        model.loss.beta=$beta \
        additional=load_models \
        load_models.kwargs.prfx=$prfx \
        load_models.plot_reps_clfs.n_mesh=50 \
        load_models.plot_reps_clfs.n_max_scatter=500 \
        load_models.plot_reps_clfs.get_title="loglike" \
        load_models.plot_reps_clfs.is_invert_yaxis=True \
        load_models.plot_reps_clfs.is_plot_test=True
done
done
done
done
done



wait 

params="col_val_subset.data=[bincifar100]
col_val_subset.datasize=[small,mini]
col_val_subset.lr=[5e-5]
col_val_subset.model=[cdib,cdibS,cdibL,vib,stochasticErm]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[2]
col_val_subset.enc_zy_nhid=[64] 
col_val_subset.beta=[0.1,1] 
"


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.style=beta \
       plot_aux_trnsf.row=data \
       plot_generalization.is_trnsf=True \
       plot_generalization.x=model \
       plot_generalization.row=data \
       plot_generalization.col=beta \
       plot_histories.col=model \
       plot_histories.row=data \
       plot_histories.style=beta \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params=$params" col_val_subset.clf_nhid=[64] 
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_metrics.x=model \
       plot_metrics.style=beta \
       plot_metrics.row=data \
       plot_generalization.is_trnsf=False \
       plot_generalization.x=model \
       plot_generalization.row=data \
       plot_generalization.col=beta \
       plot_histories.col=model \
       plot_histories.row=data \
       plot_histories.style=beta \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_generalization,plot_histories] 




