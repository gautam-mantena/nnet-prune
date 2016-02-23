// nnetbin/nnet-prune.cc

// Copyright 2015 National University of Singapore (Author: Gautam Mantena)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-affine-transform.h"


#include <sstream>
#include <string>
#include <iostream>


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
        "Perform forward pass through Neural Network.\n"
        "\n"
        "Usage:  nnet-prune [options] <model-in> <entrop-rspecifier> <model-out>\n"
        "e.g.: \n"
        "nnet-prune --tau=0.4 nnet.in ark:entropy-vectors.ark nnet.out\n";

    ParseOptions po(usage);

    double tau=0;
    po.Register("tau",&tau,"Tau values are in percentage. E.g. tau=0.4 => choose 40% of nodes.");
    
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_ifilename = po.GetArg(1),
      entropy_rspecifier = po.GetArg(2),
      model_ofilename = po.GetArg(3);


    std::string use_gpu="yes";
        
    //Select the GPU
#if HAVE_CUDA==1
    
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    //decide the threshold
    double thres=0.0;

    if (tau < 0.0 || tau > 1.0) {
      KALDI_ERR << "Tau value is not in the range of [0,1]\n";
      return 0;
    }

    {

      std::vector<double> all_entropy;      
      SequentialBaseFloatVectorReader entropy_reader(entropy_rspecifier);

      for (; !entropy_reader.Done(); entropy_reader.Next()) {
        Vector<BaseFloat> entropy_vec = entropy_reader.Value();
        for (int i = 0; i < entropy_vec.Dim(); i++ ) {
          all_entropy.push_back(entropy_vec(i));
         }
      }
      
      std::sort(all_entropy.begin(), all_entropy.end());
      int per_index = int(all_entropy.size() * tau);
      thres = all_entropy[per_index];      
    }

    Nnet nnet,onnet;
    nnet.Read(model_ifilename);

    //gathering all the components of the nnet
    std::vector<Component*> nnet_components;
    for (int i = 0; i < nnet.NumComponents(); i++) {
      nnet_components.push_back(nnet.GetComponent(i).Copy());
    }


    int no_components = nnet.NumComponents();
    int layerNo = -2; //makes things easier :)

    SequentialBaseFloatVectorReader entropy_reader(entropy_rspecifier);
    Vector<BaseFloat> prev_vec;

    for (; !entropy_reader.Done(); entropy_reader.Next()) {
      std::string layerid = entropy_reader.Key();
      Vector<BaseFloat> entropy_vec = entropy_reader.Value();

      layerNo += 2;


      if (layerNo == 0) {
        AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet_components[layerNo]);
        Matrix<BaseFloat> wt(aff_t->GetLinearity());
        Vector<BaseFloat> bias(aff_t->GetBias());

        int no_deleted = 0;
        for (int i = 0; i < entropy_vec.Dim(); i++) {
          if (entropy_vec(i) >= thres) {
            wt.RemoveRow(i - no_deleted);
            bias.RemoveElement(i - no_deleted);
            no_deleted++;
          }
        }

        AffineTransform* modLayer = new AffineTransform(wt.NumCols(),wt.NumRows());
        CuMatrix<BaseFloat> cu_wt(wt);
        CuVector<BaseFloat> cu_bias(bias);

        modLayer->SetLinearity(cu_wt);
        modLayer->SetBias(cu_bias);

        onnet.AppendComponent(modLayer);
        onnet.AppendComponent(new Sigmoid(bias.Dim(),bias.Dim()));
      }
      else {
        AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet_components[layerNo]);
        Matrix<BaseFloat> wt(aff_t->GetLinearity());
        Vector<BaseFloat> bias(aff_t->GetBias());

        int no_deleted = 0;
        for (int i = 0; i < entropy_vec.Dim(); i++) {
          if (entropy_vec(i) >= thres) {
            wt.RemoveRow(i - no_deleted);
            bias.RemoveElement(i - no_deleted);
            no_deleted++;
          }
        }

        wt.Transpose(); //deleting connections related to the previous layer
        no_deleted = 0;
        for (int i = 0; i < prev_vec.Dim(); i++) {
          if (prev_vec(i) >= thres) {
            wt.RemoveRow(i - no_deleted);
            no_deleted++;
          }
        }
        wt.Transpose(); //getting back the original structure

        AffineTransform* modLayer = new AffineTransform(wt.NumCols(),wt.NumRows());
        CuMatrix<BaseFloat> cu_wt(wt);
        CuVector<BaseFloat> cu_bias(bias);

        modLayer->SetLinearity(cu_wt);
        modLayer->SetBias(cu_bias);

        onnet.AppendComponent(modLayer);
        onnet.AppendComponent(new Sigmoid(bias.Dim(),bias.Dim()));

      }

      prev_vec = entropy_vec;
    }

    //fixing the last layers

    int fSoftmaxIndex = no_components - 1;
    int fLayerIndex = fSoftmaxIndex - 1;

    AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet_components[fLayerIndex]);
    Matrix<BaseFloat> wt(aff_t->GetLinearity());
    Vector<BaseFloat> bias(aff_t->GetBias());

    wt.Transpose();

    int no_deleted = 0;
    for (int i = 0; i < prev_vec.Dim(); i++) {
      if (prev_vec(i) >= thres) {
        wt.RemoveRow(i - no_deleted);
        no_deleted++;
      }
    }

    wt.Transpose(); //getting back the original structure

    AffineTransform* modLayer = new AffineTransform(wt.NumCols(),wt.NumRows());
    CuMatrix<BaseFloat> cu_wt(wt);
    CuVector<BaseFloat> cu_bias(bias);

    modLayer->SetLinearity(cu_wt);
    modLayer->SetBias(cu_bias);

    onnet.AppendComponent(modLayer);
    onnet.AppendComponent(new Softmax(bias.Dim(),bias.Dim()));

    //writing nnet back to file
    Output ko(model_ofilename, true);
    onnet.Write(ko.Stream(), true);


#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
