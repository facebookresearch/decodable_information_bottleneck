####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Practically show the existance of a Q sufficient statistic which is not Qi sufficient with Q'<Q 
# Hypothesis
#   - show the plot in 2 dimension where columns sweep over C and rows Q. For each row it should not be C-sufficient until reaches C=Q in the oclumns

experiment="Qsuf2DXL"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed
experiment=$precomputed 
dataset.kwargs.is_normalize=False
dataset.kwargs.is_augment=False
model.architecture.z_dim=2 
model.architecture.zy_kwargs.k_prune=0 
model.architecture.zy_kwargs.n_hidden_layers=1
clfs=QsufHeatmapXL
train.clf_kwargs.clean_after_run=training
model.norm_layer=identity
model.n_skip=0
train.kwargs.lr=1e-5
train.optim=adam
train.scheduling_mode=null
datasize=all
dataset.kwargs.is_random_targets=False 
clfs.is_reinitialize=False
train.monitor_best=tloss
train.trnsf_kwargs.max_epochs=300
train.clf_kwargs.max_epochs=300
hydra.launcher.time=3000
model=Qsufficiency
encoder=resnet18
dataset=bincifar100
$dev
"

kwargs_multi="
run=0,1,2
model.architecture.zy_kwargs.hidden_size=4,16,64,256,1024
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
  prfx="data$data"_"run$run"_"enc$enc"_
  echo "prfx:$prfx"

  # large mesh, decrease for speed (e.g. 20)
  python load_models.py $kwargs \
        load_models.recolt_data.encoders_param=model.architecture.zy_kwargs.hidden_size \
        load_models.recolt_data.encoders_vals=[4,16,64,256,1024] \
        load_models.recolt_data.clf_patterns=['clf_nhid_4/','clf_nhid_16/','clf_nhid_64/','clf_nhid_256/','clf_nhid_1024/'] \
        dataset=$data \
        encoder=$enc \
        run=$run \
        additional=load_models \
        load_models.plot_reps_clfs.n_mesh=50 \
        load_models.plot_reps_clfs.n_max_scatter=500 \
        load_models.plot_reps_clfs.diagonal_color=null \
        load_models.plot_reps_clfs.is_invert_yaxis=True \
        load_models.plot_reps_clfs.get_title="loglike" \
        load_models.kwargs.prfx=$prfx
done
done
done


params="col_val_subset.data=[bincifar100]
col_val_subset.datasize=[all]
col_val_subset.augment=[False]
col_val_subset.rand=[False]
col_val_subset.chckpnt=[tloss]
col_val_subset.schedule=[null]
col_val_subset.optim=[adam]
col_val_subset.lr=[1e-5]
col_val_subset.wdecay=[0]
col_val_subset.model=[Qsufficiency]
col_val_subset.dropout=[0]
col_val_subset.encoder=[resnet18]
col_val_subset.nskip=[0]
col_val_subset.zdim=[2]
col_val_subset.minimax=[0]
col_val_subset.mchead=[5]
"

params=$params" col_val_subset.enc_zy_nhid=[4,16,64,256,1024] 
col_val_subset.enc_zy_kpru=[0] 
col_val_subset.enc_zy_nlay=[1]
"

# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=enc_zy_nhid \
       plot_aux_trnsf.logbase_x=4 \
       plot_aux_trnsf.col=encoder \
       plot_aux_trnsf.xticks=[4,16,64,256,1024] \
       plot_histories.col=enc_zy_nhid \
       plot_histories.style=encoder \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]

params=$params" col_val_subset.clf_nhid=[4,16,64,256,1024] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params \
       plot_metrics.x=enc_zy_nhid \
       plot_metrics.style=clf_nhid \
       plot_metrics.col=encoder \
       plot_metrics.logbase_x=4 \
       plot_metrics.xticks=[4,16,64,256,1024] \
       plot_histories.col=enc_zy_nhid \
       plot_histories.style=encoder \
       plot_histories.folder_col=clf_nhid \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_metrics,plot_histories] 


