use candle::{
    CudaDevice, Module, Result, Tensor, WithDType, backend::BackendStorage, conv::CudnnFwdAlgo,
    cuda::CudaStorageSlice,
};
use candle_nn::VarBuilder;
use cudarc::{
    cudnn::{ConvForward, CudnnDataType, sys::cudnnConvolutionFwdAlgo_t},
    driver::{CudaSlice, CudaView},
};

use crate::asr::common::cudnn_ctx;

#[derive(Debug, Clone, Copy)]
pub struct Conv1dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv1dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv1dConfig,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: Conv1dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_channels, in_channels / config.groups, kernel_size),
            "weight",
        )?;
        let bias = match vb.get(out_channels, "bias") {
            Ok(bias) => Some(bias),
            Err(candle::Error::CannotFindTensor { .. }) => None,
            Err(e) => return Err(e),
        };

        Ok(Self {
            weight,
            bias,
            config,
        })
    }
}

impl Module for Conv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, in_channels, in_width) = x.dims3()?;
        let (out_channels, _, kernel_width) = self.weight.dims3()?;

        let out_width =
            (in_width + 2 * self.config.padding - self.config.dilation * (kernel_width - 1) - 1)
                / self.config.stride
                + 1;

        let op = Conv2dOp {
            batch: batch as _,
            in_channels: in_channels as _,
            out_channels: out_channels as _,
            in_height: 1,
            in_width: in_width as _,
            kernel_height: 1 as _,
            kernel_width: kernel_width as _,
            out_height: 1,
            out_width: out_width as _,
            padding: [0, self.config.padding as _],
            stride: [1, self.config.stride as _],
            dilation: [1, self.config.dilation as _],
            groups: self.config.groups as _,
            cudnn_fwd_algo: self.config.cudnn_fwd_algo,
        };

        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        let x = x
            .apply_op2(&self.weight, op)?
            .reshape((batch, out_channels, ()))?;

        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dConfig {
    pub padding: (usize, usize),
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: (0, 0),
            stride: (1, 1),
            dilation: (1, 1),
            groups: 1,
            cudnn_fwd_algo: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: Conv2dConfig,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        config: Conv2dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (
                out_channels,
                in_channels / config.groups,
                kernel_size.0,
                kernel_size.1,
            ),
            "weight",
        )?;
        let bias = match vb.get(out_channels, "bias") {
            Ok(bias) => Some(bias),
            Err(candle::Error::Io(_)) => None,
            Err(e) => return Err(e),
        };
        Ok(Self {
            weight,
            bias,
            config,
        })
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, in_channels, in_height, in_width) = x.dims4()?;
        let (out_channels, _, kernel_height, kernel_width) = self.weight.dims4()?;

        let out_height = (in_height + 2 * self.config.padding.0
            - self.config.dilation.0 * (kernel_height - 1)
            - 1)
            / self.config.stride.0
            + 1;
        let out_width = (in_width + 2 * self.config.padding.1
            - self.config.dilation.1 * (kernel_width - 1)
            - 1)
            / self.config.stride.1
            + 1;

        let op = Conv2dOp {
            batch: batch as _,
            in_channels: in_channels as _,
            out_channels: out_channels as _,
            in_height: in_height as _,
            in_width: in_width as _,
            kernel_height: kernel_height as _,
            kernel_width: kernel_width as _,
            out_height: out_height as _,
            out_width: out_width as _,
            padding: [self.config.padding.0 as _, self.config.padding.1 as _],
            stride: [self.config.stride.0 as _, self.config.stride.1 as _],
            dilation: [self.config.dilation.0 as _, self.config.dilation.1 as _],
            groups: self.config.groups as _,
            cudnn_fwd_algo: self.config.cudnn_fwd_algo,
        };

        let x = if x.is_contiguous() {
            x.clone()
        } else {
            x.contiguous()?
        };
        let x = x.apply_op2(&self.weight, op)?;

        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.dims1()?;
                let bias = bias.reshape((1, b, 1, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

struct Conv2dOp {
    batch: i32,
    in_channels: i32,
    out_channels: i32,
    in_height: i32,
    in_width: i32,
    kernel_height: i32,
    kernel_width: i32,
    out_height: i32,
    out_width: i32,
    padding: [i32; 2],
    stride: [i32; 2],
    dilation: [i32; 2],
    groups: i32,
    cudnn_fwd_algo: Option<CudnnFwdAlgo>,
}

impl Conv2dOp {
    fn launch_conv2d<T: CudnnDataType + WithDType>(
        &self,
        device: &CudaDevice,
        src: &CudaView<T>,
        filter: &CudaView<T>,
        dst: &mut CudaSlice<T>,
    ) -> Result<()> {
        let cudnn = cudnn_ctx::get(device)?;

        let desc = cudnn.create_conv2d::<T>(
            self.padding,
            self.stride,
            self.dilation,
            cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;

        let batch = self.batch;
        let x = cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [batch, self.in_channels, self.in_height, self.in_width],
        )?;
        let w = cudnn.create_4d_filter::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [
                self.out_channels,
                self.in_channels / self.groups,
                self.kernel_height,
                self.kernel_width,
            ],
        )?;
        let y = cudnn.create_4d_tensor::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [batch, self.out_channels, self.out_height, self.out_width],
        )?;

        let conv = ConvForward {
            conv: &desc,
            x: &x,
            w: &w,
            y: &y,
        };

        let alg = match self.cudnn_fwd_algo {
            None => conv.pick_algorithm()?,
            Some(CudnnFwdAlgo::ImplicitGemm) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            }
            Some(CudnnFwdAlgo::ImplicitPrecompGemm) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            Some(CudnnFwdAlgo::Gemm) => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
            Some(CudnnFwdAlgo::Direct) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
            }
            Some(CudnnFwdAlgo::Fft) => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT,
            Some(CudnnFwdAlgo::FftTiling) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            }
            Some(CudnnFwdAlgo::Winograd) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
            }
            Some(CudnnFwdAlgo::WinogradNonFused) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
            }
            Some(CudnnFwdAlgo::Count) => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT
            }
        };

        let workspace_size = conv.get_workspace_size(alg)?;
        let mut workspace = unsafe { device.alloc::<u8>(workspace_size)? };

        unsafe {
            conv.launch::<CudaSlice<u8>, _, _, _>(
                alg,
                Some(&mut workspace),
                (T::one(), T::zero()),
                src,
                filter,
                dst,
            )?;
        }
        Ok(())
    }
}

impl candle::CustomOp2 for Conv2dOp {
    fn name(&self) -> &'static str {
        "cudnn-conv2d"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        unimplemented!()
    }

    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        _l1: &candle::Layout,
        s2: &candle::CudaStorage,
        _l2: &candle::Layout,
    ) -> candle::Result<(candle::CudaStorage, candle::Shape)> {
        let device = s1.device().clone();

        let mut out = unsafe {
            device.alloc::<f32>(
                self.batch as usize
                    * self.out_channels as usize
                    * self.out_height as usize
                    * self.out_width as usize,
            )
        }?;

        let src = s1.as_cuda_slice()?.as_view();
        self.launch_conv2d(&device, &src, &s2.as_cuda_slice()?.as_view(), &mut out)?;
        let slice = CudaStorageSlice::F32(out);

        Ok((
            candle::CudaStorage { slice, device },
            (
                self.batch as usize,
                self.out_channels as usize,
                self.out_height as usize,
                self.out_width as usize,
            )
                .into(),
        ))
    }
}
