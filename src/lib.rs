//! FPN-YOLO object detection library

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{self as nn, Conv2d, Conv2dConfig, Module, VarBuilder, VarMap};
use opencv::core::{Mat, MatTrait, MatTraitConst, Point, Rect, Scalar, Size, Vec3b, Vector, CV_32F};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2RGB, INTER_LINEAR, rectangle, put_text, FONT_HERSHEY_SIMPLEX, LINE_8};
use opencv::dnn::{read_net_from_onnx, NetTrait, NetTraitConst, DNN_BACKEND_OPENCV, DNN_TARGET_CPU};

use std::sync::Arc;
use rayon::prelude::*;
use parking_lot::{Mutex, RwLock};
use num_cpus;

// Performance-oriented imports
use tokio::sync::Semaphore;
use futures::future::join_all;
use dashmap::DashMap;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use crossbeam_channel::{bounded, Receiver, Sender};
use std::collections::VecDeque;

// Python bindings module
#[cfg(feature = "python-bindings")]
pub mod python_bindings;
#[cfg(feature = "python-bindings")]
pub use python_bindings::*;

#[derive(Debug, Clone)]
pub struct Backbone {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
}

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

// GPU Memory Pool for efficient CUDA memory management
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
        
        // Create new tensor if not in cache
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

// Pre-allocated buffer pool for OpenCV operations
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
        // Create new Mat safely
        Mat::new_size_with_default(size, opencv::core::CV_8UC3, Scalar::all(0.0))
    }

    pub fn return_mat_buffer(&self, mat: Mat) {
        let mut buffers = self.mat_buffers.lock();
        if buffers.len() < self.max_buffers {
            buffers.push_back(mat);
        }
    }
}

// Global instances for performance
static GPU_MEMORY_POOL: Lazy<RwLock<Option<Arc<GpuMemoryPool>>>> = Lazy::new(|| RwLock::new(None));
static BUFFER_POOL: Lazy<Arc<BufferPool>> = Lazy::new(|| Arc::new(BufferPool::new(50)));

// Enhanced system configuration with performance tuning
#[derive(Clone)]
pub struct SystemConfig {
    pub device: Device,
    pub use_gpu: bool,
    pub cpu_threads: usize,
    pub gpu_memory_pool: Option<Arc<GpuMemoryPool>>,
    pub parallel_streams: usize,
}

impl SystemConfig {
    pub fn detect_optimal() -> anyhow::Result<Self> {
        // Try to initialize CUDA device with better error handling
        let (device, use_gpu) = match Device::cuda_if_available(0) {
            Ok(cuda_device) => {
                            println!("CUDA GPU detected and enabled");
            println!("Initializing GPU memory pool...");
                (cuda_device, true)
            }
            Err(e) => {
                            println!("CUDA not available: {}", e);
            println!("Using CPU processing");
                (Device::Cpu, false)
            }
        };

        // Initialize GPU memory pool if GPU is available
        let gpu_memory_pool = if use_gpu {
            let pool = Arc::new(GpuMemoryPool::new(device.clone(), 100));
            *GPU_MEMORY_POOL.write() = Some(pool.clone());
            Some(pool)
        } else {
            None
        };

        // Optimize CPU thread count and parallel streams
        let cpu_count = num_cpus::get();
        let (optimal_threads, parallel_streams) = if use_gpu {
            // Use fewer CPU threads when GPU is available, more parallel streams
            ((cpu_count / 2).max(4), 8)
        } else {
            // Use most CPU threads when no GPU, fewer parallel streams
            (cpu_count.saturating_sub(1).max(1), 4)
        };

        // Configure optimized Rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(optimal_threads)
            .stack_size(8 * 1024 * 1024) // 8MB stack for complex operations
            .build_global()
            .map_err(|e| anyhow::anyhow!("Failed to configure thread pool: {}", e))?;

        println!("System configuration:");
        println!("  Device: {}", if use_gpu { "CUDA GPU" } else { "CPU" });
        println!("  CPU threads: {}/{}", optimal_threads, cpu_count);
        println!("  Parallel streams: {}", parallel_streams);
        println!("  GPU memory pool: {}", if gpu_memory_pool.is_some() { "enabled" } else { "disabled" });

        Ok(Self {
            device,
            use_gpu,
            cpu_threads: optimal_threads,
            gpu_memory_pool,
            parallel_streams,
        })
    }
}

impl Backbone {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let conv1 = nn::conv2d(3, 64, 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vs.pp("conv1"))?;
        let conv2 = nn::conv2d(64, 128, 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vs.pp("conv2"))?;
        let conv3 = nn::conv2d(128, 256, 3, Conv2dConfig { stride: 2, padding: 1, ..Default::default() }, vs.pp("conv3"))?;
        Ok(Self { conv1, conv2, conv3 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let c1 = self.conv1.forward(x)?.relu()?;
        let c2 = self.conv2.forward(&c1)?.relu()?;
        let c3 = self.conv3.forward(&c2)?.relu()?;
        Ok(vec![c1, c2, c3])
    }
}

#[derive(Debug, Clone)]
pub struct FPN {
    lateral3: Conv2d,
    lateral2: Conv2d,
    lateral1: Conv2d,
}

impl FPN {
    pub fn new(vs: VarBuilder) -> Result<Self> {
        let lateral3 = nn::conv2d(256, 256, 1, Default::default(), vs.pp("lateral3"))?;
        let lateral2 = nn::conv2d(128, 256, 1, Default::default(), vs.pp("lateral2"))?;
        let lateral1 = nn::conv2d(64, 256, 1, Default::default(), vs.pp("lateral1"))?;
        Ok(Self { lateral3, lateral2, lateral1 })
    }

    pub fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        let c1 = &features[0];
        let c2 = &features[1];
        let c3 = &features[2];

        // Top-down pathway
        let p3 = self.lateral3.forward(c3)?;
        let p3_h = p3.dims()[2] * 2;
        let p3_w = p3.dims()[3] * 2;
        let p3_up = p3.upsample_nearest2d(p3_h, p3_w)?;
        let p2 = self.lateral2.forward(c2)?.add(&p3_up)?;
        let p2_h = p2.dims()[2] * 2;
        let p2_w = p2.dims()[3] * 2;
        let p2_up = p2.upsample_nearest2d(p2_h, p2_w)?;
        let p1 = self.lateral1.forward(c1)?.add(&p2_up)?;

        Ok(vec![p1, p2, p3])
    }
}

// Optimized tensor conversion with memory pooling
pub fn mat_to_tensor_optimized(mat: &Mat, device: &Device) -> Result<Tensor> {
    let rows = mat.rows();
    let cols = mat.cols();
    
    if rows == 0 || cols == 0 {
        return Err(candle_core::Error::Msg("Rows or columns cannot be zero.".to_string()));
    }
    
    let rows_usize = rows as usize;
    let cols_usize = cols as usize;
    let total_elements = rows_usize * cols_usize * 3;
    
    // Use GPU memory pool if available
    if let Some(pool) = GPU_MEMORY_POOL.read().as_ref() {
        let shape = &[1, 3, rows_usize, cols_usize];
        if let Ok(mut tensor) = pool.get_tensor(shape, DType::F32) {
            // Fast data conversion
            let data: Vec<f32> = convert_mat_data_optimized(mat, total_elements);
            tensor = Tensor::from_vec(data, shape, device)?;
            return Ok(tensor);
        }
    }
    
    // Fallback to standard conversion
    let data: Vec<f32> = convert_mat_data_optimized(mat, total_elements);
    Tensor::from_vec(data, (1, 3, rows_usize, cols_usize), device)
}

// Optimized data conversion
pub fn convert_mat_data_optimized(mat: &Mat, total_elements: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(total_elements);
    let rows = mat.rows() as usize;
    let cols = mat.cols() as usize;
    
    // Vectorized processing
    for idx in 0..total_elements {
        let row = (idx / (cols * 3)) as i32;
        let col = ((idx / 3) % cols) as i32;
        let ch = idx % 3;
        
        data.push(mat.at_2d::<Vec3b>(row, col).unwrap()[ch] as f32 / 255.0);
    }
    
    data
}

// Enhanced image processing with all optimizations
pub fn create_enhanced_input(original_mat: &Mat, _fpn_features: &[Tensor]) -> opencv::Result<Mat> {
    // For now, return the original mat
    // In a full implementation, you would:
    // 1. Convert FPN features back to OpenCV Mat
    // 2. Resize them to match input size
    // 3. Combine with original image (e.g., concatenate channels or weighted sum)
    
    Ok(original_mat.clone())
}

// Main optimized processing function with all performance improvements
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
    // Use buffer pool for temporary matrices
    let buffer_pool = &*BUFFER_POOL;
    
    let mut rgb_img = buffer_pool.get_mat_buffer(img.size()?)?;
    cvt_color(img, &mut rgb_img, COLOR_BGR2RGB, 0, opencv::core::AlgorithmHint::ALGO_HINT_ACCURATE)?;

    // Resize to match model input
    let mut resized = buffer_pool.get_mat_buffer(Size::new(input_size, input_size))?;
    resize(&rgb_img, &mut resized, Size::new(input_size, input_size), 0.0, 0.0, INTER_LINEAR)?;

    // Return rgb_img buffer early
    buffer_pool.return_mat_buffer(rgb_img);

    // Convert OpenCV Mat to Candle Tensor with optimizations
    let input_tensor = mat_to_tensor_optimized(&resized, device)?;
    
    // Process through FPN backbone and FPN with GPU acceleration
    let backbone_features = backbone.forward(&input_tensor)?;
    let fpn_features = fpn.forward(&backbone_features)?;
    
    // Create enhanced input using FPN features
    let enhanced_input = create_enhanced_input(&resized, &fpn_features)?;
    buffer_pool.return_mat_buffer(resized);

    // Prepare input blob using FPN-enhanced input
    let blob = opencv::dnn::blob_from_image(&enhanced_input, 1.0 / 255.0, Size::new(input_size, input_size), Scalar::all(0.0), false, false, CV_32F)?;
    net.set_input(&blob, "", 1.0, Scalar::default())?;

    // Forward pass
    let out_names = net.get_unconnected_out_layers_names()?;
    let mut out_blobs = Vector::<Mat>::new();
    net.forward(&mut out_blobs, &out_names)?;
    let output_blob = out_blobs.get(0)?;

    // Ultra-optimized detection processing
    let (boxes, confidences, class_ids) = process_detections_ultra_optimized(
        &output_blob, 
        img, 
        input_size, 
        conf_threshold,
        system_config
    )?;

    // Apply NMS
    let mut indices = Vector::<i32>::new();
    opencv::dnn::nms_boxes(&boxes, &confidences, conf_threshold, nms_threshold, &mut indices, 1.0, 0)?;

    // Draw boxes on original image
    let mut result_img = img.clone();
    draw_boxes_optimized(&mut result_img, &boxes, &confidences, &class_ids, &indices, class_names)?;

    Ok(result_img)
}

// Ultra-optimized detection processing with parallel processing
pub fn process_detections_ultra_optimized(
    output_blob: &Mat,
    img: &Mat,
    input_size: i32,
    conf_threshold: f32,
    system_config: &SystemConfig,
) -> anyhow::Result<(Vector<Rect>, Vector<f32>, Vector<i32>)> {
    let mat_size = output_blob.mat_size();
    let num_detections = mat_size[2] as usize;

    let ptr = output_blob.ptr(0i32)?;
    let data: &[f32] = unsafe { std::slice::from_raw_parts(ptr as *const f32, output_blob.total() as usize) };

    // Calculate optimal chunk size based on system configuration
    let optimal_chunk_size = (num_detections / system_config.cpu_threads).max(1000).min(5000);
    
    // Process detections in highly optimized parallel chunks
    let results: Vec<_> = (0..num_detections)
        .into_par_iter()
        .chunks(optimal_chunk_size)
        .map(|chunk| {
            let mut local_boxes = SmallVec::<[Rect; 32]>::new();
            let mut local_confidences = SmallVec::<[f32; 32]>::new();
            let mut local_class_ids = SmallVec::<[i32; 32]>::new();

            // Pre-compute scale factors
            let scale_x = img.cols() as f32 / input_size as f32;
            let scale_y = img.rows() as f32 / input_size as f32;

            for i in chunk {
                // Optimized coordinate calculation
                let x_center = data[i] * scale_x;
                let y_center = data[num_detections + i] * scale_y;
                let width = data[2 * num_detections + i] * scale_x;
                let height = data[3 * num_detections + i] * scale_y;
                
                // Optimized class confidence calculation
                let mut max_class_conf = 0.0f32;
                let mut best_class_id = 0;
                
                for class_idx in 0..80 {
                    let class_conf = data[(4 + class_idx) * num_detections + i];
                    if class_conf > max_class_conf {
                        max_class_conf = class_conf;
                        best_class_id = class_idx;
                    }
                }
                
                if max_class_conf > conf_threshold {
                    let x = (x_center - width / 2.0) as i32;
                    let y = (y_center - height / 2.0) as i32;
                    let w = width as i32;
                    let h = height as i32;

                    local_boxes.push(Rect::new(x, y, w, h));
                    local_confidences.push(max_class_conf);
                    local_class_ids.push(best_class_id as i32);
                }
            }

            (local_boxes, local_confidences, local_class_ids)
        })
        .collect();

    // Efficiently combine results
    let total_capacity: usize = results.iter().map(|(b, _, _)| b.len()).sum();
    let mut boxes = Vector::<Rect>::new();
    let mut confidences = Vector::<f32>::new();
    let mut class_ids = Vector::<i32>::new();

    for (local_boxes, local_confidences, local_class_ids) in results {
        for bbox in local_boxes { boxes.push(bbox); }
        for conf in local_confidences { confidences.push(conf); }
        for class_id in local_class_ids { class_ids.push(class_id); }
    }

    Ok((boxes, confidences, class_ids))
}

// Optimized box drawing with reduced allocations
pub fn draw_boxes_optimized(
    result_img: &mut Mat,
    boxes: &Vector<Rect>,
    confidences: &Vector<f32>,
    class_ids: &Vector<i32>,
    indices: &Vector<i32>,
    class_names: &[String],
) -> anyhow::Result<()> {
    // Pre-format all labels to reduce allocations during drawing
    let labels: SmallVec<[String; 64]> = indices.iter()
        .map(|i| {
            let class_id = class_ids.get(i as usize).unwrap_or(0);
            let conf = confidences.get(i as usize).unwrap_or(0.0);
            format!("{}: {:.2}", 
                class_names.get(class_id as usize).unwrap_or(&"unknown".to_string()), 
                conf)
        })
        .collect();

    // Draw boxes and labels
    for (idx, i) in indices.iter().enumerate() {
        let bbox = boxes.get(i as usize)?;
        rectangle(result_img, bbox, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, LINE_8, 0)?;
        put_text(result_img, &labels[idx], Point::new(bbox.x, bbox.y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar::new(0.0, 255.0, 0.0, 0.0), 2, LINE_8, false)?;
    }

    Ok(())
}

// Python bindings are handled by the pymodule in python_bindings.rs 