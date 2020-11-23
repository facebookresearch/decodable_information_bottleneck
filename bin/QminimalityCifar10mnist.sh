####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : 
# - look at the effect of Q minimality on : training loss, mnist, worst case test loss, test loss

experiment="QminimalityCifar10mnist"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar10mnist
dataset.kwargs.is_augment=False 
hydra.launcher.time=2000
model.Q_zy.hidden_size=128
model.Q_zy.n_hidden_layers=1
model=cdib
encoder=mlpl
model.architecture.z_dim=1024
datasize=all
datasize.max_epochs=200
model.loss.altern_minimax=5
model.loss.n_per_head=1
model=cdib
model.loss.is_higher=True
model.architecture.is_wrap_batchnorm=True
model.loss.z_norm_reg=0
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0,0.01,0.1,1,10,100
"

if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &

    wait 

    PYTHON_CODE=$(cat <<END
import glob, shutil
old = "default"
pattern = "/private/home/yannd/projects/Decodable_Information_Bottleneck/tmp_results/QminimalityCifar10mnist/**/clfs_{}/**/transformer".format(old)
folders=glob.glob(pattern, recursive=True)

for new in ["avg","distractor","worst","random"]:
    for folder in folders: 
        new_folder = folder.replace(old, new)
        try:
            shutil.copytree(folder, new_folder)
        except FileExistsError:
            shutil.rmtree(new_folder)
            shutil.copytree(folder, new_folder)
END
)

    res="$(python -c "$PYTHON_CODE")"

    wait

    python main.py $kwargs $kwargs_multi $kwargs2 clfs="distractor","avg","worst" -m &

    sleep 2m 
    
  done
fi

wait 

params="col_val_subset.data=[cifar10mnist]
col_val_subset.datasize=[all]
col_val_subset.model=[cdib]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0,0.01,0.1,1,10,100]
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
       col_val_subset.clfs=["default"] \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=True \
       plot_generalization.logbase_x=10 \
       plot_generalization.col=datasize \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.logbase_x=10 \
       plot_aux_trnsf.col=datasize \
       plot_histories.row=datasize \
       plot_histories.col=beta \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]

params_clf=$params" col_val_subset.clf_nhid=[128] 
col_val_subset.clf_kpru=[0] 
col_val_subset.clf_nlay=[1]
col_val_subset.clfs=[distractor,default,avg,worst]
"

# CLASSIFIER
python aggregate.py \
       experiment=$experiment \
       $params_clf \
       kwargs.pretty_renamer.Default=Worst_Case_Label \
       plot_generalization.x=beta \
       plot_generalization.is_trnsf=False \
       plot_generalization.logbase_x=10 \
       plot_generalization.col=datasize \
       plot_generalization.style=clfs \
       plot_generalization.is_legend_out=True \
       plot_generalization.is_no_legend_title=False \
       plot_generalization.set_kwargs.xlim=[0,1e2] \
       plot_metrics.x=beta \
       plot_metrics.logbase_x=10 \
       plot_metrics.col=datasize \
       plot_metrics.style=clfs \
       plot_histories.row=datasize \
       plot_histories.col=beta \
       plot_histories.row=clfs \
       recolt_data.pattern_histories="tmp_results/$experiment/**/clf_nhid_*/**/last_epoch_history.json" \
       recolt_data.pattern_aux_trnsf=null \
       mode=[save_tables,plot_generalization,plot_metrics,plot_histories]