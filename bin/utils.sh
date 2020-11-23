####
#Copyright (c) Facebook, Inc. and its affiliates.
#
#This source code is licensed under the MIT license found in the
#LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

run="0"
dev=""
prfx=""
run="0,1,2,3,4"
time="2880" #2 days
is_plot_only=false


# DEV MODE ?
while getopts ':dnsi:p:l' flag; do
  case "${flag}" in
    d ) 
      dev='train.trnsf_kwargs.max_epochs=3 train.clf_kwargs.max_epochs=3 datasize.max_epochs=3 hydra.launcher.partition=dev is_nvidia_smi=True' #hydra.verbose=true 
      time="20"
      prfx="dev_"
      run="0"
      echo "Dev mode ..."
      ;;
    n ) 
      dev='datasize.max_epochs=3' 
      time="50"
      prfx="nano_"
      run="0"
      echo "Nano mode ..."
      ;;
    s ) 
      dev='datasize.max_epochs=200' 
      time="800"
      prfx="small_"
      run="0,1,2"
      echo "Small mode ..."
      ;;
    l ) 
      dev='' 
      prfx="large_"
      run="0,1,2,3,4,5,6,7,8,9"
      echo "Large mode ..."
      ;;
    p ) 
      is_plot_only=true
      prfx=${OPTARG}
      echo "Plotting only ..."
      ;;
    i ) 
      dev="hydra.launcher.partition=priority hydra.launcher.params.queue_parameters.slurm.comment=${OPTARG}"
      echo "Priority mode : ${OPTARG}..."
      ;;
    \? ) 
      echo "Usage: "$experiment".sh [-dnspl]" 
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done

experiment="$prfx""$experiment"

results="tmp_results/$experiment"
if [ -d "$results" ]; then

  echo -n "$results exist. Should I delete it (y/n) ? "
  read answer

  if [ "$answer" != "${answer#[Yy]}" ] ;then
      echo "Deleted $results"
      rm -rf $results
  fi
fi  