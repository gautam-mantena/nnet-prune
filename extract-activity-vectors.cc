// nnetbin/extract-activity-vectors.cc

// Copyright 2015 National University of Singapore (Authors: Gautam Mantena, Sim Khe Chai)

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
        "Extract activity vectors for each of the nodes of a Neural Network.\n"
        "\n"
      "Usage:  extract-activity-vectors [options] <model-in> <feature-rspecifier> <alignment-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        "extract-activation-vectors --S=40 --buffer-index=1 nnet ark:features.ark ark:monophones.ali ark:activations.ark\n";

    ParseOptions po(usage);

    int S=40;
    po.Register("S",&S,"Number of attributes for which activation vectors are computed.");

    int buffer_index=1;
    po.Register("buffer-index",&buffer_index,"This is the id of the lowest alignment id.");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format).");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA."); 

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;


    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
      feature_rspecifier = po.GetArg(2),
      align_rspecifier = po.GetArg(3),
      feature_wspecifier = po.GetArg(4);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);

    // disable dropout,
    nnet_transf.SetDropoutRetention(1.0);
    nnet.SetDropoutRetention(1.0);


    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialInt32VectorReader align_reader(align_rspecifier);    
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Vector<BaseFloat> scounts(S);
    scounts.SetZero();

    int32 num_done = 0;
    std::vector<Matrix<BaseFloat> > activity_vectors((nnet.NumComponents() - 2)/2);
    //initializing the activity vectors
    for (int i = 1; i < nnet.NumComponents() - 2; i = i + 2) { // avoid the softmax from the last layer.
      Component::ComponentType nnet_type = nnet.GetComponent(i).GetType();
      if (nnet_type != Component::kSigmoid) {
        KALDI_ERR << "Component " << i << " is not Sigmoid.\n";
        return 0;
      }
      AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet.GetComponent(i-1).Copy());
      int noNodes = aff_t->GetBias().Dim();
      int layerNo = (i-1)/2;
      activity_vectors[layerNo].Resize(noNodes,S);
      activity_vectors[layerNo].SetZero();
    }
    //iterating each alignment
    for ( ; !align_reader.Done(); align_reader.Next() ) {
      std::string utt = align_reader.Key();
      const std::vector<int32> &align_vec = align_reader.Value();


      if (!feature_reader.HasKey(utt)) {
        KALDI_LOG << "Utt: " << utt << " not available\n";
        continue;
      }
      Matrix<BaseFloat> mat = feature_reader.Value(utt);

      //checking features
      if (!KALDI_ISFINITE(mat.Sum())) {
        KALDI_ERR << "NaN or inf found in features of " << utt << "\n";
        return 0;
      }

      //push the features onto gpu
      feats = mat;
      nnet_transf.Feedforward(feats, &feats_transf);

      //checking the transfer features
      if (!KALDI_ISFINITE(feats_transf.Sum())) {
        KALDI_ERR << "NaN or inf found (some issue with splicing) in features of " << utt << "\n";
        return 0;
      }

      nnet.Propagate(feats_transf,&nnet_out);
      std::vector<CuMatrix<BaseFloat> > layer_activations = nnet.PropagateBuffer();

      if (align_vec.size() != mat.NumRows()) {
        KALDI_ERR << "The number of frames are not same for features and alignment file for the utt " << utt;
        return 0;
      }


      Matrix<BaseFloat> G(mat.NumRows(), S);
      for (int i = 0; i < mat.NumRows(); i++) {
        Vector<BaseFloat> temp(S);
        temp.SetZero();
        if (align_vec[i] - buffer_index >= S) {
          KALDI_ERR << "The ids in the alignment exceeds S in the utt " << utt;
          return 0;
        }
        scounts(align_vec[i] - buffer_index) += 1; //this is to normalize G so that one attribute does not dominate the others
        temp(align_vec[i] - buffer_index) = 1;
        G.CopyRowFromVec(temp,i);
        
      }

      for (int noc = 2; noc < layer_activations.size() - 2; noc = noc + 2) {
        //0th index contains the input feats, 1th contains the affine transform, 2nd contains the sigmoid.
        int layerNo = (noc - 2)/2;
        Matrix<BaseFloat> H(layer_activations[noc].NumRows(), layer_activations[noc].NumCols());
        H.CopyFromMat(layer_activations[noc]);
        activity_vectors[layerNo].AddMatMat(1, H, kTrans, G, kNoTrans, 1);
      }
      num_done++;
    }

    if (num_done == 0) {
      KALDI_ERR << "No utts processed\n";
      return 0;
    }

    //normalizing the activity vectors
    for (int layerNo = 0; layerNo < activity_vectors.size(); layerNo++) {
      Matrix<BaseFloat> out_mat(activity_vectors[layerNo].NumRows(), activity_vectors[layerNo].NumCols());
      out_mat.SetZero();
      
      for (int i = 0; i < activity_vectors[layerNo].NumRows(); i++) {
        Vector<BaseFloat> row(S);
        row.CopyRowFromMat(activity_vectors[layerNo],i);
        row.DivElements(scounts);
        row.Scale(1.0/row.Sum());
        out_mat.CopyRowFromVec(row,i);
      }
      std::ostringstream temp;
      temp << layerNo + 1;
      feature_writer.Write(temp.str(), out_mat);
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files";

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
