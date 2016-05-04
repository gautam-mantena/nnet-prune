#!/bin/bash

. ./path.sh

feat_transform=kaldi-data-models/tri3a_dnn/final.feature_transform
dbn_model=kaldi-data-models/tri3a_dnn/final.nnet
dir=data-files

#extract-activity-vectors --use-gpu="yes" --S=40 --buffer-index=0 \
extract-activity-vectors --S=40 --buffer-index=0 \
  --feature-transform=$feat_transform $dbn_model \
  scp:$dir/dev_feats.scp ark:$dir/dev_0330.monophones.alignment ark,t:activation-vectors.ark || exit 1


#S - represents the number of attributes
#buffer-index - each index number is given an id. If the monophones start with id=0 then the buffer-index=0 or it starts with the index with starting monophone 



compute-entropy-from-activations ark:activation-vectors.ark ark,t:entropy.txt || exit 1

