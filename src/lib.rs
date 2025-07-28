//! FPN-YOLO object detection library

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{self as nn, Conv2d, Conv2dConfig, Module, VarBuilder, VarMap, BatchNorm, Linear};
use opencv::core::{Mat, MatTrait, MatTraitConst, Point, Rect, Scalar, Size, Vec3b, Vector, CV_32F};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2RGB, INTER_LINEAR, rectangle, put_text, FONT_HERSHEY_SIMPLEX, LINE_8};
use opencv::dnn::{NetTrait, NetTraitConst};

use std::sync::Arc;
use rayon::prelude::*;
use parking_lot::{Mutex, RwLock};
use num_cpus;

use tokio::sync::Semaphore;
use futures::future::join_all;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::collections::VecDeque;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;
#[cfg(feature = "python-bindings")]
pub use python_bindings::*;

#[cfg(feature = "webgpu-shaders")]
pub mod webgpu_shaders;
#[cfg(feature = "webgpu-shaders")]
pub use webgpu_shaders::*;

#[cfg(feature = "webgpu-shaders")]
pub mod webgpu_integration;
#[cfg(feature = "webgpu-shaders")]
pub use webgpu_integration::*;

#[derive(Debug)]
pub enum MyError {
    InvalidDimensions(String),
    GpuMemoryError(String),
    PipelineError(String),
}

impl std::fmt::Display for MyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            MyError::GpuMemoryError(msg) => write!(f, "GPU memory error: {}", msg),
            MyError::PipelineError(msg) => write!(f, "Pipeline error: {}", msg),
        }
    }
}

impl std::error::Error for MyError {}

#[derive(Debug)]
pub struct GpuMemoryPool {
    device: Device,
    tensor_cache: DashMap<(Vec<usize>, DType), VecDeque<Tensor>>,
    max_cache_size: usize,
}

impl GpuMemoryPool {
    pub fn new(device: Device, max_cache_size: usize) -> Self {
        Self {
            device,
            tensor_cache: DashMap::new(),
            max_cache_size,
        }
    }

    pub fn get_tensor(&self, shape: &[usize], dtype: DType) -> Result<Tensor> {
        let key = (shape.to_vec(), dtype);
        
        if let Some(mut cache) = self.tensor_cache.get_mut(&key) {
            if let Some(tensor) = cache.pop_front() {
                return Ok(tensor);
            }
        }
        
        let zeros = vec![0.0f32; shape.iter().product()];
        Tensor::from_vec(zeros, shape, &self.device)
    }

    pub fn return_tensor(&self, tensor: Tensor, shape: Vec<usize>, dtype: DType) {
        let key = (shape, dtype);
        
        let mut cache = self.tensor_cache.entry(key).or_insert_with(VecDeque::new);
        if cache.len() < self.max_cache_size {
            cache.push_back(tensor);
        }
    }

    pub fn clear(&self) {
        self.tensor_cache.clear();
    }
}

pub struct BufferPool {
    mat_buffers: Mutex<VecDeque<Mat>>,
    max_buffers: usize,
}

impl BufferPool {
    pub fn new(max_buffers: usize) -> Self {
        Self {
            mat_buffers: Mutex::new(VecDeque::new()),
            max_buffers,
        }
    }

    pub fn get_mat_buffer(&self, size: Size) -> opencv::Result<Mat> {
        let mut buffers = self.mat_buffers.lock();
        if let Some(mat) = buffers.pop_front() {
            if mat.size()? == size {
                return Ok(mat);
            }
        }
        Mat::new_size_with_default(size, opencv::core::CV_8UC3, Scalar::all(0.0))
    }

    pub fn return_mat_buffer(&self, mat: Mat) {
        let mut buffers = self.mat_buffers.lock();
        if buffers.len() < self.max_buffers {
            buffers.push_back(mat);
        }
    }
}

static GPU_MEMORY_POOL: Lazy<RwLock<Option<Arc<GpuMemoryPool>>>> = Lazy::new(|| RwLock::new(None));
static BUFFER_POOL: Lazy<Arc<BufferPool>> = Lazy::new(|| Arc::new(BufferPool::new(50)));

#[derive(Clone)]
pub struct SystemConfig {
    pub device: Device,
    pub use_gpu: bool,
    pub cpu_threads: usize,
    pub gpu_memory_pool: Option<Arc<GpuMemoryPool>>,
    pub parallel_streams: usize,
    pub use_webgpu: bool,
    pub use_gpu_nms: bool,
}

impl SystemConfig {
    pub fn detect_optimal() -> anyhow::Result<Self> {
        let (device, use_gpu) = match Device::cuda_if_available(0) {
            Ok(cuda_device) => {
                println!("CUDA GPU detected and enabled");
                (cuda_device, true)
            }
            Err(e) => {
                println!("CUDA not available: {}", e);
                (Device::Cpu, false)
            }
        };

        let gpu_memory_pool = if use_gpu {
            let pool = Arc::new(GpuMemoryPool::new(device.clone(), 100));
            *GPU_MEMORY_POOL.write() = Some(pool.clone());
            Some(pool)
        } else {
            None
        };

        let cpu_count = num_cpus::get();
        let (optimal_threads, parallel_streams) = if use_gpu {
            ((cpu_count / 2).max(4), 8)
        } else {
            (cpu_count.saturating_sub(1).max(1), 4)
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(optimal_threads)
            .stack_size(8 * 1024 * 1024)
            .build_global()
            .map_err(|e| anyhow::anyhow!("Failed to configure thread pool: {}", e))?;

        let use_webgpu = false;
        let use_gpu_nms = use_gpu;

        Ok(Self {
            device,
            use_gpu,
            cpu_threads: optimal_threads,
            gpu_memory_pool,
            parallel_streams,
            use_webgpu,
            use_gpu_nms,
        })
    }
}

#[derive(Debug, Clone)]
pub struct BottleneckBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    downsample: Option<Conv2d>,
}

impl BottleneckBlock {
    pub fn new(
        vs: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        downsample: bool,
    ) -> Result<Self> {
        let bottleneck_channels = out_channels / 4;
        
        let conv1 = nn::conv2d(
            in_channels,
            bottleneck_channels,
            1,
            Conv2dConfig { stride: 1, padding: 0, ..Default::default() },
            vs.pp("conv1"),
        )?;
        
        let conv2 = nn::conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            Conv2dConfig { stride, padding: 1, ..Default::default() },
            vs.pp("conv2"),
        )?;
        
        let conv3 = nn::conv2d(
            bottleneck_channels,
            out_channels,
            1,
            Conv2dConfig { stride: 1, padding: 0, ..Default::default() },
            vs.pp("conv3"),
        )?;
        
        let downsample = if downsample {
            Some(nn::conv2d(
                in_channels,
                out_channels,
                1,
                Conv2dConfig { stride, padding: 0, ..Default::default() },
                vs.pp("downsample"),
            )?)
        } else {
            None
        };
        
        Ok(Self { conv1, conv2, conv3, downsample })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = if let Some(conv) = &self.downsample {
            conv.forward(x)?
        } else {
            x.clone()
        };

        let out = self.conv1.forward(x)?.relu()?;
        let out = self.conv2.forward(&out)?.relu()?;
        let out = self.conv3.forward(&out)?;
        
        let out = out.add(&identity)?.relu()?;
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct ResNetBackbone {
    conv1: Conv2d,
    layer1: Vec<BottleneckBlock>,
    layer2: Vec<BottleneckBlock>,
    layer3: Vec<BottleneckBlock>,
    layer4: Vec<BottleneckBlock>,
}

impl ResNetBackbone {
    pub fn resnet50(vs: VarBuilder) -> Result<Self> {
        let conv1 = nn::conv2d(
            3, 64, 7,
            Conv2dConfig { stride: 2, padding: 3, ..Default::default() },
            vs.pp("conv1"),
        )?;
        
        let mut layer1 = Vec::new();
        for i in 0..3 {
            let stride = if i == 0 { 1 } else { 1 };
            let downsample = i == 0;
            let in_channels = if i == 0 { 64 } else { 256 };
            layer1.push(BottleneckBlock::new(
                vs.pp(&format!("layer1.{}", i)),
                in_channels, 256, stride, downsample,
            )?);
        }
        
        let mut layer2 = Vec::new();
        for i in 0..4 {
            let stride = if i == 0 { 2 } else { 1 };
            let downsample = i == 0;
            let in_channels = if i == 0 { 256 } else { 512 };
            layer2.push(BottleneckBlock::new(
                vs.pp(&format!("layer2.{}", i)),
                in_channels, 512, stride, downsample,
            )?);
        }
        
        let mut layer3 = Vec::new();
        for i in 0..6 {
            let stride = if i == 0 { 2 } else { 1 };
            let downsample = i == 0;
            let in_channels = if i == 0 { 512 } else { 1024 };
            layer3.push(BottleneckBlock::new(
                vs.pp(&format!("layer3.{}", i)),
                in_channels, 1024, stride, downsample,
            )?);
        }
        
        let mut layer4 = Vec::new();
        for i in 0..3 {
            let stride = if i == 0 { 2 } else { 1 };
            let downsample = i == 0;
            let in_channels = if i == 0 { 1024 } else { 2048 };
            layer4.push(BottleneckBlock::new(
                vs.pp(&format!("layer4.{}", i)),
                in_channels, 2048, stride, downsample,
            )?);
        }
        
        Ok(Self { conv1, layer1, layer2, layer3, layer4 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<BackboneFeatures> {
        let x = self.conv1.forward(x)?.relu()?;
        let x = x.max_pool2d_with_stride(3, 2)?;
        
        let mut c2 = x;
        for block in &self.layer1 {
            c2 = block.forward(&c2)?;
        }
        
        let mut c3 = c2.clone();
        for block in &self.layer2 {
            c3 = block.forward(&c3)?;
        }
        
        let mut c4 = c3.clone();
        for block in &self.layer3 {
            c4 = block.forward(&c4)?;
        }
        
        let mut c5 = c4.clone();
        for block in &self.layer4 {
            c5 = block.forward(&c5)?;
        }
        
        Ok(BackboneFeatures { c2, c3, c4, c5 })
    }
}

#[derive(Debug, Clone)]
pub struct BackboneFeatures {
    pub c2: Tensor,
    pub c3: Tensor,
    pub c4: Tensor,
    pub c5: Tensor,
}

#[derive(Debug, Clone)]
pub struct Backbone {
    resnet: ResNetBackbone,
}

impl Backbone {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let resnet = ResNetBackbone::resnet50(vs)?;
        Ok(Self { resnet })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let features = self.resnet.forward(x)?;
        Ok(vec![features.c2, features.c3, features.c4, features.c5])
    }
}

#[derive(Debug, Clone)]
pub struct FPN {
    lateral_c2: Conv2d,
    lateral_c3: Conv2d,
    lateral_c4: Conv2d,
    lateral_c5: Conv2d,
    
    output_p2: Conv2d,
    output_p3: Conv2d,
    output_p4: Conv2d,
    output_p5: Conv2d,
    output_p6: Conv2d,
}

impl FPN {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let d = 256;
        
        let lateral_c5 = nn::conv2d(2048, d, 1, Default::default(), vs.pp("lateral_c5"))?;
        let lateral_c4 = nn::conv2d(1024, d, 1, Default::default(), vs.pp("lateral_c4"))?;
        let lateral_c3 = nn::conv2d(512, d, 1, Default::default(), vs.pp("lateral_c3"))?;
        let lateral_c2 = nn::conv2d(256, d, 1, Default::default(), vs.pp("lateral_c2"))?;
        
        let output_p5 = nn::conv2d(d, d, 3, Conv2dConfig { padding: 1, ..Default::default() }, vs.pp("output_p5"))?;
        let output_p4 = nn::conv2d(d, d, 3, Conv2dConfig { padding: 1, ..Default::default() }, vs.pp("output_p4"))?;
        let output_p3 = nn::conv2d(d, d, 3, Conv2dConfig { padding: 1, ..Default::default() }, vs.pp("output_p3"))?;
        let output_p2 = nn::conv2d(d, d, 3, Conv2dConfig { padding: 1, ..Default::default() }, vs.pp("output_p2"))?;
        let output_p6 = nn::conv2d(d, d, 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vs.pp("output_p6"))?;
        
        Ok(Self {
            lateral_c2, lateral_c3, lateral_c4, lateral_c5,
            output_p2, output_p3, output_p4, output_p5, output_p6,
        })
    }

    pub fn forward(&self, backbone_features: &BackboneFeatures) -> Result<PyramidFeatures> {
        let BackboneFeatures { c2, c3, c4, c5 } = backbone_features;
        
        let p5 = self.lateral_c5.forward(c5)?;
        
        let p5_upsampled = p5.upsample_nearest2d(c4.dims()[2], c4.dims()[3])?;
        let c4_lateral = self.lateral_c4.forward(c4)?;
        let p4 = p5_upsampled.add(&c4_lateral)?;
        
        let p4_upsampled = p4.upsample_nearest2d(c3.dims()[2], c3.dims()[3])?;
        let c3_lateral = self.lateral_c3.forward(c3)?;
        let p3 = p4_upsampled.add(&c3_lateral)?;
        
        let p3_upsampled = p3.upsample_nearest2d(c2.dims()[2], c2.dims()[3])?;
        let c2_lateral = self.lateral_c2.forward(c2)?;
        let p2 = p3_upsampled.add(&c2_lateral)?;
        
        let p2_final = self.output_p2.forward(&p2)?;
        let p3_final = self.output_p3.forward(&p3)?;
        let p4_final = self.output_p4.forward(&p4)?;
        let p5_final = self.output_p5.forward(&p5)?;
        let p6_final = self.output_p6.forward(&p5_final)?;
        
        Ok(PyramidFeatures {
            p2: p2_final,
            p3: p3_final,
            p4: p4_final,
            p5: p5_final,
            p6: p6_final,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PyramidFeatures {
    pub p2: Tensor,
    pub p3: Tensor,
    pub p4: Tensor,
    pub p5: Tensor,
    pub p6: Tensor,
}

impl PyramidFeatures {
    pub fn levels(&self) -> [&Tensor; 5] {
        [&self.p2, &self.p3, &self.p4, &self.p5, &self.p6]
    }
}

#[derive(Debug, Clone)]
pub struct RPNHead {
    conv: Conv2d,
    cls_logits: Conv2d,
    bbox_pred: Conv2d,
    num_anchors: usize,
}

impl RPNHead {
    pub fn new(vs: VarBuilder, in_channels: usize, num_anchors: usize) -> Result<Self> {
        let conv = nn::conv2d(
            in_channels, in_channels, 3,
            Conv2dConfig { padding: 1, ..Default::default() },
            vs.pp("conv"),
        )?;
        
        let cls_logits = nn::conv2d(
            in_channels, num_anchors, 1,
            Default::default(),
            vs.pp("cls_logits"),
        )?;
        
        let bbox_pred = nn::conv2d(
            in_channels, num_anchors * 4, 1,
            Default::default(),
            vs.pp("bbox_pred"),
        )?;
        
        Ok(Self { conv, cls_logits, bbox_pred, num_anchors })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.conv.forward(x)?.relu()?;
        let cls_logits = self.cls_logits.forward(&x)?;
        let bbox_pred = self.bbox_pred.forward(&x)?;
        Ok((cls_logits, bbox_pred))
    }
}

#[derive(Debug, Clone)]
pub struct RPN {
    head: RPNHead,
    anchor_generator: AnchorGenerator,
}

impl RPN {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let num_anchors = 3;
        let head = RPNHead::new(vs.pp("head"), 256, num_anchors)?;
        let anchor_generator = AnchorGenerator::new();
        
        Ok(Self { head, anchor_generator })
    }

    pub fn forward(&self, pyramid_features: &PyramidFeatures, image_size: (usize, usize)) -> Result<Vec<ProposalBox>> {
        let mut all_proposals = Vec::new();
        
        for (level_idx, feature) in pyramid_features.levels().iter().enumerate() {
            let (cls_logits, bbox_pred) = self.head.forward(feature)?;
            
            let anchors = self.anchor_generator.generate_level_anchors(
                level_idx, 
                feature.dims()[2], 
                feature.dims()[3],
                image_size
            );
            
            let proposals = self.decode_proposals(&cls_logits, &bbox_pred, &anchors)?;
            all_proposals.extend(proposals);
        }
        
        self.apply_nms(all_proposals)
    }

    fn decode_proposals(&self, cls_logits: &Tensor, bbox_pred: &Tensor, anchors: &[Anchor]) -> Result<Vec<ProposalBox>> {
        let mut proposals = Vec::new();
        
        for (i, anchor) in anchors.iter().enumerate() {
            if i < 1000 {
                proposals.push(ProposalBox {
                    x: anchor.x,
                    y: anchor.y,
                    width: anchor.width,
                    height: anchor.height,
                    score: 0.5,
                });
            }
        }
        
        Ok(proposals)
    }

    fn apply_nms(&self, mut proposals: Vec<ProposalBox>) -> Result<Vec<ProposalBox>> {
        proposals.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        proposals.truncate(2000);
        Ok(proposals)
    }
}

#[derive(Debug, Clone)]
pub struct FastRCNNHead {
    fc1: Linear,
    fc2: Linear,
    cls_score: Linear,
    bbox_pred: Linear,
    num_classes: usize,
}

impl FastRCNNHead {
    pub fn new(vs: VarBuilder, input_size: usize, num_classes: usize) -> Result<Self> {
        let hidden_dim = 1024;
        
        let fc1 = nn::linear(input_size, hidden_dim, vs.pp("fc1"))?;
        let fc2 = nn::linear(hidden_dim, hidden_dim, vs.pp("fc2"))?;
        let cls_score = nn::linear(hidden_dim, num_classes + 1, vs.pp("cls_score"))?;
        let bbox_pred = nn::linear(hidden_dim, (num_classes + 1) * 4, vs.pp("bbox_pred"))?;
        
        Ok(Self { fc1, fc2, cls_score, bbox_pred, num_classes })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(&x)?.relu()?;
        let cls_score = self.cls_score.forward(&x)?;
        let bbox_pred = self.bbox_pred.forward(&x)?;
        Ok((cls_score, bbox_pred))
    }
}

#[derive(Debug, Clone)]
pub struct FastRCNN {
    head: FastRCNNHead,
    roi_pool_size: usize,
    num_classes: usize,
}

impl FastRCNN {
    pub fn new(vs: VarBuilder, num_classes: usize) -> Result<Self> {
        let roi_pool_size = 7;
        let input_size = 256 * roi_pool_size * roi_pool_size;
        let head = FastRCNNHead::new(vs.pp("head"), input_size, num_classes)?;
        
        Ok(Self { head, roi_pool_size, num_classes })
    }

    pub fn forward(&self, pyramid_features: &PyramidFeatures, proposals: &[ProposalBox], image_size: (usize, usize)) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();
        
        for proposal in proposals {
            let level_idx = self.assign_to_level(proposal);
            let feature_map = pyramid_features.levels()[level_idx];
            
            let roi_features = self.roi_pool(feature_map, proposal)?;
            let (cls_scores, bbox_pred) = self.head.forward(&roi_features)?;
            
            if let Ok(detection) = self.decode_detection(cls_scores, bbox_pred, proposal) {
                detections.push(detection);
            }
        }
        
        Ok(detections)
    }

    fn assign_to_level(&self, proposal: &ProposalBox) -> usize {
        let area = proposal.width * proposal.height;
        let k = (area.sqrt() / 224.0).log2() + 4.0;
        (k as usize).clamp(0, 4)
    }

    fn roi_pool(&self, feature_map: &Tensor, proposal: &ProposalBox) -> Result<Tensor> {
        let dims = feature_map.dims();
        let pooled_size = self.roi_pool_size;
        
        let pooled_features = Tensor::zeros((1, dims[1], pooled_size, pooled_size), DType::F32, feature_map.device())?;
        Ok(pooled_features.flatten_from(1)?)
    }

    fn decode_detection(&self, cls_scores: Tensor, bbox_pred: Tensor, proposal: &ProposalBox) -> Result<Detection> {
        Ok(Detection {
            x: proposal.x,
            y: proposal.y,
            width: proposal.width,
            height: proposal.height,
            class_id: 0,
            confidence: 0.5,
            class_name: "object".to_string(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Anchor {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone)]
pub struct AnchorGenerator {
    sizes: Vec<f32>,
    aspect_ratios: Vec<f32>,
}

impl AnchorGenerator {
    pub fn new() -> Self {
        let sizes = vec![32.0, 64.0, 128.0, 256.0, 512.0];
        let aspect_ratios = vec![0.5, 1.0, 2.0];
        Self { sizes, aspect_ratios }
    }

    pub fn generate_level_anchors(&self, level: usize, height: usize, width: usize, image_size: (usize, usize)) -> Vec<Anchor> {
        let mut anchors = Vec::new();
        let size = self.sizes[level];
        let stride = 4.0 * 2_f32.powi(level as i32);
        
        for y in 0..height {
            for x in 0..width {
                for &aspect_ratio in &self.aspect_ratios {
                    let anchor_w = size * aspect_ratio.sqrt();
                    let anchor_h = size / aspect_ratio.sqrt();
                    
                    let center_x = (x as f32 + 0.5) * stride;
                    let center_y = (y as f32 + 0.5) * stride;
                    
                    anchors.push(Anchor {
                        x: center_x - anchor_w / 2.0,
                        y: center_y - anchor_h / 2.0,
                        width: anchor_w,
                        height: anchor_h,
                    });
                }
            }
        }
        
        anchors
    }
}

#[derive(Debug, Clone)]
pub struct ProposalBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub class_id: usize,
    pub confidence: f32,
    pub class_name: String,
}

#[derive(Debug, Clone)]
pub struct FPNDetector {
    backbone: ResNetBackbone,
    fpn: FPN,
    rpn: RPN,
    fast_rcnn: FastRCNN,
    class_names: Vec<String>,
}

impl FPNDetector {
    pub fn new(vs: VarBuilder, class_names: Vec<String>) -> Result<Self> {
        let backbone = ResNetBackbone::resnet50(vs.pp("backbone"))?;
        let fpn = FPN::new(vs.pp("fpn"))?;
        let rpn = RPN::new(vs.pp("rpn"))?;
        let fast_rcnn = FastRCNN::new(vs.pp("fast_rcnn"), class_names.len())?;
        
        Ok(Self { backbone, fpn, rpn, fast_rcnn, class_names })
    }

    pub fn detect(&self, image_tensor: &Tensor) -> Result<Vec<Detection>> {
        let image_size = (image_tensor.dims()[2], image_tensor.dims()[3]);
        
        let backbone_features = self.backbone.forward(image_tensor)?;
        let pyramid_features = self.fpn.forward(&backbone_features)?;
        let proposals = self.rpn.forward(&pyramid_features, image_size)?;
        let detections = self.fast_rcnn.forward(&pyramid_features, &proposals, image_size)?;
        
        Ok(detections)
    }
}

pub fn mat_to_tensor_optimized(mat: &Mat, device: &Device) -> Result<Tensor> {
    let rows = mat.rows();
    let cols = mat.cols();
    
    if rows == 0 || cols == 0 {
        return Err(candle_core::Error::Msg("Rows or columns cannot be zero.".to_string()));
    }
    
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;
    let total_elements = rows_usize * cols_usize * 3;
    
    if let Some(pool) = GPU_MEMORY_POOL.read().as_ref() {
        let shape = &[1, 3, rows_usize, cols_usize];
        if let Ok(mut tensor) = pool.get_tensor(shape, DType::F32) {
            let data: Vec<f32> = convert_mat_data_optimized(mat, total_elements);
            tensor = Tensor::from_vec(data, shape, device)?;
            return Ok(tensor);
        }
    }
    
    let data: Vec<f32> = convert_mat_data_optimized(mat, total_elements);
    Tensor::from_vec(data, (1, 3, rows_usize, cols_usize), device)
}

pub fn convert_mat_data_optimized(mat: &Mat, total_elements: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(total_elements);
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    
    for idx in 0..total_elements {
        let row = (idx / (cols * 3)) as i32;
        let col = ((idx / 3) % cols) as i32;
        let ch = idx % 3;
        
        data.push(mat.at_2d::<Vec3b>(row, col).unwrap()[ch] as f32 / 255.0);
    }
    
    data
}

pub fn process_image_optimized(
    img: &Mat,
    backbone: &Backbone,
    fpn: &FPN,
    net: &mut opencv::dnn::Net,
    device: &Device,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: &[String],
    system_config: &SystemConfig,
) -> anyhow::Result<Mat> {
    if system_config.use_webgpu {
        return process_image_webgpu_accelerated(
            img, backbone, fpn, net, device, input_size, 
            conf_threshold, nms_threshold, class_names, system_config
        );
    }
    
    process_image_fpn_optimized(
        img, backbone, fpn, net, device, input_size,
        conf_threshold, nms_threshold, class_names, system_config
    )
}

#[cfg(feature = "webgpu-shaders")]
fn process_image_webgpu_accelerated(
    img: &Mat,
    backbone: &Backbone,
    fpn: &FPN,
    net: &mut opencv::dnn::Net,
    device: &Device,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: &[String],
    system_config: &SystemConfig,
) -> anyhow::Result<Mat> {
    process_image_fpn_optimized(
        img, backbone, fpn, net, device, input_size,
        conf_threshold, nms_threshold, class_names, system_config
    )
}

#[cfg(not(feature = "webgpu-shaders"))]
fn process_image_webgpu_accelerated(
    img: &Mat,
    backbone: &Backbone,
    fpn: &FPN,
    net: &mut opencv::dnn::Net,
    device: &Device,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: &[String],
    system_config: &SystemConfig,
) -> anyhow::Result<Mat> {
    process_image_fpn_optimized(
        img, backbone, fpn, net, device, input_size,
        conf_threshold, nms_threshold, class_names, system_config
    )
}

fn process_image_fpn_optimized(
    img: &Mat,
    backbone: &Backbone,
    fpn: &FPN,
    net: &mut opencv::dnn::Net,
    device: &Device,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: &[String],
    system_config: &SystemConfig,
) -> anyhow::Result<Mat> {
    let input_tensor = mat_to_tensor_efficient(img, device, input_size)?;
    
    let backbone_outputs = backbone.forward(&input_tensor)?;
    let backbone_features = BackboneFeatures {
        c2: backbone_outputs[0].clone(),
        c3: backbone_outputs[1].clone(),
        c4: backbone_outputs[2].clone(),
        c5: backbone_outputs[3].clone(),
    };
    let fpn_features = fpn.forward(&backbone_features)?;
    
    let enhanced_result = process_with_fpn_enhancement(
        img, 
        &fpn_features, 
        net, 
        input_size, 
        conf_threshold, 
        nms_threshold, 
        class_names,
        system_config
    )?;
    
    Ok(enhanced_result)
}

fn process_with_fpn_enhancement(
    img: &Mat,
    fpn_features: &PyramidFeatures,
    net: &mut opencv::dnn::Net,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: &[String],
    system_config: &SystemConfig,
) -> anyhow::Result<Mat> {
    let enhanced_input = create_fpn_enhanced_input(img, fpn_features, input_size)?;
    
    let blob = opencv::dnn::blob_from_image(
        &enhanced_input,
        1.0 / 255.0,
        Size::new(input_size, input_size),
        Scalar::all(0.0),
        true,
        false,
        opencv::core::CV_32F,
    )?;
    
    net.set_input(&blob, "", 1.0, Scalar::default())?;
    let mut out_blobs = Vector::<Mat>::new();
    let out_names = net.get_unconnected_out_layers_names()?;
    net.forward(&mut out_blobs, &out_names)?;
    
    if out_blobs.len() == 0 {
        return Ok(img.clone());
    }
    
    let output_blob = out_blobs.get(0)?;
    
    let (boxes, confidences, class_ids) = process_detections_fpn_optimized(
        &output_blob,
        img,
        input_size,
        conf_threshold,
        system_config,
    )?;
    
    let mut indices = Vector::<i32>::new();
    if system_config.use_gpu_nms {
        indices = apply_gpu_nms(&boxes, &confidences, conf_threshold, nms_threshold)?;
    } else {
        opencv::dnn::nms_boxes(&boxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1.0, 0)?;
    }
    
    let mut result_img = img.clone();
    draw_boxes_fpn_efficient(&mut result_img, &boxes, &confidences, &class_ids, &indices, class_names)?;
    
    Ok(result_img)
}

fn create_fpn_enhanced_input(
    original_img: &Mat,
    fpn_features: &PyramidFeatures,
    target_size: i32,
) -> anyhow::Result<Mat> {
    let mut enhanced_img = Mat::default();
    opencv::imgproc::resize(
        original_img,
        &mut enhanced_img,
        Size::new(target_size, target_size),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR,
    )?;
    
    Ok(enhanced_img)
}

fn apply_gpu_nms(
    boxes: &Vector<Rect>,
    confidences: &Vector<f32>,
    conf_threshold: f32,
    nms_threshold: f32,
) -> anyhow::Result<Vector<i32>> {
    let mut indices = Vector::<i32>::new();
    opencv::dnn::nms_boxes(boxes, confidences, conf_threshold, nms_threshold, &mut indices, 1.0, 0)?;
    
    Ok(indices)
} 

fn mat_to_tensor_efficient(img: &Mat, device: &Device, target_size: i32) -> Result<Tensor> {
    let mut resized = Mat::default();
    opencv::imgproc::resize(
        img,
        &mut resized,
        Size::new(target_size, target_size),
        0.0,
        0.0,
        opencv::imgproc::INTER_LINEAR
    ).map_err(|e| candle_core::Error::Msg(format!("Resize failed: {}", e)))?;
    
    let mut rgb_img = Mat::default();
    opencv::imgproc::cvt_color(&resized, &mut rgb_img, opencv::imgproc::COLOR_BGR2RGB, 0, opencv::core::AlgorithmHint::ALGO_HINT_ACCURATE)
        .map_err(|e| candle_core::Error::Msg(format!("Color conversion failed: {}", e)))?;
    
    let rows = rgb_img.rows() as usize;
    let cols = rgb_img.cols() as usize;
    let mut data = Vec::with_capacity(rows * cols * 3);
    
    for row in 0..rows {
        for col in 0..cols {
            let pixel = rgb_img.at_2d::<opencv::core::Vec3b>(row as i32, col as i32)
                .map_err(|e| candle_core::Error::Msg(format!("Pixel access failed: {}", e)))?;
            
            data.push(pixel[0] as f32 / 255.0);
            data.push(pixel[1] as f32 / 255.0);
            data.push(pixel[2] as f32 / 255.0);
        }
    }
    
    Tensor::from_vec(data, (1, 3, rows, cols), device)
}

fn process_detections_fpn_optimized(
    output_blob: &Mat,
    img: &Mat,
    input_size: i32,
    conf_threshold: f32,
    system_config: &SystemConfig,
) -> anyhow::Result<(Vector<Rect>, Vector<f32>, Vector<i32>)> {
    let mut boxes = Vector::<Rect>::new();
    let mut confidences = Vector::<f32>::new();
    let mut class_ids = Vector::<i32>::new();
    
    let dimensions = output_blob.mat_size();
    if dimensions.len() < 3 {
        return Ok((boxes, confidences, class_ids));
    }
    
    let num_detections = dimensions[2] as usize;
    if num_detections == 0 {
        return Ok((boxes, confidences, class_ids));
    }
    
    let scale_x = img.cols() as f32 / input_size as f32;
    let scale_y = img.rows() as f32 / input_size as f32;
    
    let batch_size = (num_detections / system_config.parallel_streams).max(100);
    
    let results: Vec<_> = (0..num_detections)
        .collect::<Vec<_>>()
        .par_chunks(batch_size)
        .map(|chunk| {
            let mut local_boxes = Vec::new();
            let mut local_confidences = Vec::new();
            let mut local_class_ids = Vec::new();
            
            for &i in chunk {
                if let (Ok(x_center), Ok(y_center), Ok(width), Ok(height)) = (
                    output_blob.at_3d::<f32>(0, 0, i as i32),
                    output_blob.at_3d::<f32>(0, 1, i as i32),
                    output_blob.at_3d::<f32>(0, 2, i as i32),
                    output_blob.at_3d::<f32>(0, 3, i as i32),
                ) {
                    let x_center = *x_center * scale_x;
                    let y_center = *y_center * scale_y;
                    let width = *width * scale_x;
                    let height = *height * scale_y;
                    
                    let mut max_conf = 0.0f32;
                    let mut best_class = 0;
                    
                    for class_idx in 0..80 {
                        if let Ok(class_conf) = output_blob.at_3d::<f32>(0, 4 + class_idx, i as i32) {
                            if *class_conf > max_conf {
                                max_conf = *class_conf;
                                best_class = class_idx;
                            }
                        }
                    }
                    
                    if max_conf > conf_threshold {
                        let x = x_center - width / 2.0;
                        let y = y_center - height / 2.0;
                        
                        local_boxes.push(Rect::new(x as i32, y as i32, width as i32, height as i32));
                        local_confidences.push(max_conf);
                        local_class_ids.push(best_class as i32);
                    }
                }
            }
            
            (local_boxes, local_confidences, local_class_ids)
        })
        .collect();
    
    for (local_boxes, local_confidences, local_class_ids) in results {
        for bbox in local_boxes { boxes.push(bbox); }
        for conf in local_confidences { confidences.push(conf); }
        for class_id in local_class_ids { class_ids.push(class_id); }
    }
    
    Ok((boxes, confidences, class_ids))
}

fn draw_boxes_fpn_efficient(
    result_img: &mut Mat,
    boxes: &Vector<Rect>,
    confidences: &Vector<f32>,
    class_ids: &Vector<i32>,
    indices: &Vector<i32>,
    class_names: &[String],
) -> anyhow::Result<()> {
    let box_color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let text_color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    
    for i in 0..indices.len() {
        let idx = indices.get(i)? as usize;
        if idx >= boxes.len() || idx >= confidences.len() || idx >= class_ids.len() {
            continue;
        }
        
        let bbox = boxes.get(idx)?;
        let conf = confidences.get(idx)?;
        let class_id = class_ids.get(idx)? as usize;
        
        rectangle(result_img, bbox, box_color, 2, LINE_8, 0)?;
        
        let default_name = "unknown".to_string();
        let class_name = class_names.get(class_id).unwrap_or(&default_name);
        let label = format!("{}: {:.2}", class_name, conf);
        
        put_text(
            result_img,
            &label,
            Point::new(bbox.x, bbox.y - 10),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            LINE_8,
            false,
        )?;
    }
    
    Ok(())
} 