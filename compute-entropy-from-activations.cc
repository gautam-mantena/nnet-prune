// featbin/compute-entropy-from-activations.cc

// Copyright 2015 National University of Singapore (author: Gautam  Mantena)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Computes entropy from the activation vectors\n"
        "Usage: compute-entropy-from-activations [options] <activationVectors-rspecifier> <feats-wspecifier>\n"
        "e.g.: compute-entropy-from-activations scp:activationVectors.scp ark,scp:feats.ark,feats.scp\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string acc_rspec = po.GetArg(1),
      feat_wspec = po.GetArg(2);


    int32 num_utt_done = 0;

    SequentialBaseFloatMatrixReader acc_reader(acc_rspec);
    BaseFloatVectorWriter feat_writer(feat_wspec);

    for ( ; !acc_reader.Done(); acc_reader.Next()) {
      std::string uttid = acc_reader.Key();
      Matrix<BaseFloat> acc_mat = acc_reader.Value();

      Vector<BaseFloat> entropy(acc_mat.NumRows());

      for (int i = 0; i < acc_mat.NumRows(); i++) {
        Vector<BaseFloat> acc_vec(acc_mat.NumCols());
        Vector<BaseFloat> acc_lvec(acc_mat.NumCols());

        acc_vec.CopyRowFromMat(acc_mat,i);
        acc_lvec.CopyRowFromMat(acc_mat,i);
        for (int k=0; k < acc_lvec.Dim(); k++) {
          if (acc_lvec(k) == 0) {
            acc_lvec(k) = 1.0e-20;
          }          
        }
        acc_lvec.ApplyLog();
        acc_vec.MulElements(acc_lvec);

        double sum = acc_vec.Sum();
        sum = sum/(-1.0 * log(acc_mat.NumCols()));
        entropy(i) = sum;
      }

      if (entropy.Min() < 0 || entropy.Max() > 1) {
        KALDI_ERR << "Entropy value is not in the limits of [0,1]: " << uttid << "\n";
        return 0;
      }

      feat_writer.Write(uttid, entropy); 
      num_utt_done++;
            
    }
    
    KALDI_LOG << "Utterances Processed: " << num_utt_done;
    return (num_utt_done != 0 ? 0 : 1);


  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


