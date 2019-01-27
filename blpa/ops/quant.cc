// --Ayan Chakrabarti <ayan@wustl.edu>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;

typedef float float32;
typedef unsigned char uint8;

// Save ACT
void gpuSave(const GPUDevice& d, uint8* lhs,
	     const float32 * rhs, const float32 * bias,
	     int numEl, int numCh, int nBits);

class SaveActGPU : public OpKernel {
public:
  explicit SaveActGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,context->GetAttr("nbits", &nBits_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& rhs_ = context->input(1);
    const Tensor& bias_ = context->input(2);
    context->forward_ref_input_to_ref_output(0, 0);
    Tensor lhs_ = context->mutable_input(0,false);

    const float32 * rhs, * bias; uint8 * lhs;

    rhs = rhs_.flat<float32>().data();
    bias = bias_.flat<float32>().data();
    lhs = (uint8*) lhs_.flat<float32>().data();

    int numEl = rhs_.shape().num_elements(), numCh = bias_.shape().num_elements();


    gpuSave(context->eigen_device<GPUDevice>(),lhs,rhs,bias,numEl,numCh,nBits_);
  }

private:
  int nBits_;
};


REGISTER_OP("SaveAct")
.Input("var: Ref(float32)")
.Input("act: float32")
.Input("bias: float32")
.Attr("nbits: int")
.Output("out_var: Ref(float32)")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0,c->input(1));
    return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("SaveAct").Device(DEVICE_GPU), SaveActGPU);


// Restore ACT
void gpuRest(const GPUDevice& d, float32 * act, float32 * Rm,
	     const uint8 * var, const float32 * bias,
	     int numEl, int numCh, int nBits);

class RestActGPU : public OpKernel {
public:
  explicit RestActGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,context->GetAttr("nbits", &nBits_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& var_ = context->input(0);
    const Tensor& bias_ = context->input(1);

    TensorShape oshp = var_.shape();
    oshp.set_dim(3, bias_.shape().num_elements());

    Tensor *act_ ,*Rm_;

    OP_REQUIRES_OK(context,context->allocate_output(0,oshp,&act_));
    OP_REQUIRES_OK(context,context->allocate_output(1,oshp,&Rm_));

    
    const float32 * bias; const uint8 * var;
    float32 * act, * Rm;

    bias = bias_.flat<float32>().data();
    var = (uint8*) var_.flat<float32>().data();
    act = act_->flat<float32>().data();
    Rm = Rm_->flat<float32>().data();
    
    int numEl = oshp.num_elements(), numCh = bias_.shape().num_elements();


    gpuRest(context->eigen_device<GPUDevice>(),act,Rm,var,bias,numEl,numCh,nBits_);
  }

private:
  int nBits_;
};


REGISTER_OP("RestAct")
.Input("var: float32")
.Input("bias: float32")
.Attr("nbits: int")
.Output("act: float32")
.Output("rm: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle v, b, out;
    v = c->input(0); b = c->input(1);
    c->ReplaceDim(v,3,c->NumElements(b),&out);
    c->set_output(0,out); c->set_output(1,out);
    return Status::OK();
  });

REGISTER_KERNEL_BUILDER(Name("RestAct").Device(DEVICE_GPU), RestActGPU);
