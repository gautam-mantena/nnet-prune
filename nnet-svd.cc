// nnetbin/nnet-svd.cc

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
#include "nnet/nnet-linear-transform.h"
#include "matrix/kaldi-matrix.h"

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
        "Usage:  nnet-svd [options] <model-in> <model-out>\n"
        "e.g.: \n"
        "nnet-svd --Rp=0.9 nnet.in nnet.out\n";

    ParseOptions po(usage);

    double Rp=0;
    po.Register("Rp",&Rp,"Rp values are in percentage. E.g. Rp=0.9 => 0.9 * min(N_l-1, N_l).");
    
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_ifilename = po.GetArg(1),
      model_ofilename = po.GetArg(2);

    std::string use_gpu="yes";
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    //decide the threshold
    Nnet nnet,onnet;
    nnet.Read(model_ifilename);

    if (Rp < 0.0 || Rp > 1.0) {
      KALDI_ERR << "Rp value is not in the range of [0,1]\n";
      return 0;
    }

    {
      //adding the transformation to the first layer
      //onnet.AppendComponent(nnet.GetComponent(0).Copy());
    }

    for (int i = 0; i < nnet.NumComponents(); i++) { //do not want to perform any SVD at the input layer. Not any more
      Component::ComponentType compType = nnet.GetComponent(i).GetType();
      if (compType != Component::kAffineTransform) {
        onnet.AppendComponent(nnet.GetComponent(i).Copy());        
        continue;
      }
      AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet.GetComponent(i).Copy());
      Matrix<BaseFloat> wt(aff_t->GetLinearity());

      //performing SVD
      //for SVD to work in Kaldi, the number rows >= num cols
      int trans_flag = 0;
      if (wt.NumRows() < wt.NumCols()) {
        wt.Transpose();
        trans_flag = 1;
      }
      //std::cout << wt.NumRows() << " " << wt.NumCols() << " " << trans_flag << std::endl;


      int minDim = std::min(wt.NumRows(), wt.NumCols());
      Matrix<BaseFloat> U(wt.NumRows(), minDim), Vt(minDim, minDim);
      Vector<BaseFloat> S(minDim);
      wt.DestructiveSvd(&S, &U, &Vt);
      SortSvd(&S, &U, &Vt);

      Matrix<BaseFloat> U_k, Vt_k;
      Vector<BaseFloat> S_k;
      int k = (wt.NumRows() * wt.NumCols())/(wt.NumRows() + wt.NumCols());
      k = k * Rp;


      //std::cout << "U " << U.NumRows() << " " << U.NumCols() << std::endl;
      //std::cout << "Vt " << Vt.NumRows() << " " << Vt.NumCols() << std::endl;
      //std::cout << "K " << k << std::endl;

      if (trans_flag == 0) {
        U_k.Resize(wt.NumRows(),k);
        Vt_k.Resize(k, wt.NumCols());
        S_k.Resize(k);

        //copying data
        U_k.CopyFromMat(U.Range(0,U_k.NumRows(), 0, U_k.NumCols()));
        Vt_k.CopyFromMat(Vt.Range(0,Vt_k.NumRows(), 0, Vt_k.NumCols()));
        S_k.CopyFromVec(S.Range(0,k));
      }
      else {
        wt.Transpose();
        U_k.Resize(wt.NumRows(), k);
        Vt_k.Resize(k, wt.NumCols());
        S_k.Resize(k);

        U.Transpose();
        Vt.Transpose();

        U_k.CopyFromMat(Vt.Range(0,U_k.NumRows(), 0, U_k.NumCols()));
        Vt_k.CopyFromMat(U.Range(0,Vt_k.NumRows(), 0, Vt_k.NumCols()));
        S_k.CopyFromVec(S.Range(0,k));
      }

      //generating the transformation
      S_k.ApplyPow(0.5);
      Matrix<BaseFloat> Up(wt.NumRows(),k);
      Up.SetZero();
      Up.AddMatDiagVec(1.0, U_k, kNoTrans, S_k);

      Matrix<BaseFloat> Lp(k, wt.NumCols());
      Lp.SetZero();
      Lp.AddDiagVecMat(1.0, S_k, Vt_k, kNoTrans);

      {
        //Adding Transformation
        LinearTransform* newLayer = new LinearTransform(Lp.NumCols(), Lp.NumRows());
        CuMatrix<BaseFloat> cu_wt(Lp);
        //CuVector<BaseFloat> cu_bias(Lp.NumRows());
        //cu_bias.SetZero();
        newLayer->SetLinearity(cu_wt);
        //newLayer->SetBias(cu_bias);
        onnet.AppendComponent(newLayer);
      }

      {
        AffineTransform* modLayer = new AffineTransform(Up.NumCols(), Up.NumRows());
        CuMatrix<BaseFloat> cu_wt(Up);
        
        AffineTransform* aff_t = dynamic_cast<AffineTransform*>(nnet.GetComponent(i).Copy());
        CuVector<BaseFloat> cu_bias(aff_t->GetBias());
        modLayer->SetLinearity(cu_wt);
        modLayer->SetBias(cu_bias);
        onnet.AppendComponent(modLayer);
      }
    }

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
