####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

# Goal : Compare effect of deterministic and stochastic cdib

experiment="RegularizerDet"

source `dirname $0`/utils.sh

#precomputed="$prfx""precomputed"
precomputed=$experiment

kwargs="trnsf_experiment=$precomputed 
experiment=$precomputed 
encoder=mlpl
dataset=cifar10
dataset.kwargs.is_augment=False 
train.kwargs.lr=5e-5
model.architecture.z_dim=1024
hydra.launcher.time=500
datasize.max_epochs=200
$dev
"

kwargs_multi="
run=0,1,2
model.loss.beta=0.01,0.1,1,10
model=simplecdib,simplecdibdet
"


if [ "$is_plot_only" = false ] ; then
  for kwargs1 in  ""
  do

    # precompute the transformer if not already done
    python main.py is_precompute_trnsf=True $kwargs $kwargs_multi $kwargs1 -m &
      
    
  done
fi

wait 

params="col_val_subset.data=[cifar10]
col_val_subset.lr=[5e-5]
col_val_subset.model=[simplecdib,simplecdibdet]
col_val_subset.encoder=[mlpl]
col_val_subset.zdim=[1024]
col_val_subset.beta=[0.01,0.1,1,10]
"


# ENCODER
python aggregate.py \
       experiment=$precomputed \
       save_experiment="$experiment"/trnsf \
       $params \
       plot_aux_trnsf.x=beta \
       plot_aux_trnsf.style=model \
       plot_aux_trnsf.logbase_x=10 \
       plot_generalization.x=beta \
       plot_generalization.style=model \
       plot_generalization.logbase_x=10 \
       plot_generalization.is_trnsf=True \
       plot_histories.col=beta \
       plot_histories.style=model \
       recolt_data.pattern_results=null \
       mode=[save_tables,plot_aux_trnsf,plot_generalization,plot_histories]
