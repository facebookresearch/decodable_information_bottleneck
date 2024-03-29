####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : See how well we do as a regularizer 

experiment="Regularizer"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
dataset=cifar10
dataset.kwargs.is_augment=False 
train.kwargs.lr=1e-2
model.architecture.z_dim=1024
hydra.launcher.time=500
datasize.max_epochs=200
$dev
"

kwargs_multi="
run=0,1,2
model=dropout,vib,simplecdib01,simplecdib,erm,cdib,wdecay,simplecdibdet,vcdib,simplevcdib
encoder=resnet18,mlpl
datasize=all
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 train.monitor_best=last -m &

    wait

    PYTHON_CODE=$(cat <<END
import glob, shutil
old = "last"
pattern = "/private/home/yannd/projects/Decodable_Information_Bottleneck/tmp_results/Regularizer/**/chckpnt_{}/**/transformer".format(old)
folders=glob.glob(pattern, recursive=True)

for new in ["vloss"]:
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

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 train.monitor_best=vloss train.trnsf_kwargs.is_train=False -m &
      
    
  done
fi

wait 

params="col_val_subset.data=[cifar10]
col_val_subset.lr=[1e-2]
col_val_subset.encoder=[mlpl,resnet18]
col_val_subset.zdim=[1024]
"


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=model \
       plot_aux_trnsf.row=data \
       plot_aux_trnsf.col=encoder \
       plot_aux_trnsf.folder_col=chckpnt \
       plot_aux_trnsf.folder_col=chckpnt \
       plot_aux_trnsf.x_rotate=45 \
       plot_histories.col=model \
       plot_histories.row=data \
       plot_histories.style=encoder \
       plot_histories.folder_col=chckpnt \
       plot_histories.x_rotate=45 \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_histories]
