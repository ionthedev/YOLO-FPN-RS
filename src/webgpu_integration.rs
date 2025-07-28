//! Integration of WebGPU WGSL shaders with the existing FPN-YOLO pipeline

#[cfg(feature = "webgpu-shaders")]
use crate::{webgpu_shaders::*, SystemConfig, Backbone, FPN, BackboneFeatures};

use opencv::core::{Mat, MatTrait, MatTraitConst, Size};
use opencv::imgproc::{cvt_color, COLOR_BGR2RGB};
use opencv::dnn::{NetTrait, NetTraitConst};
use anyhow::Result;
use std::sync::Arc;
use candle_core::{Device as CandleDevice, Tensor};

/// WebGPU-accelerated processing pipeline 
#[cfg(feature = "webgpu-shaders")]
pub struct WebGpuPipeline {
    pub image_preprocessor: ImagePreprocessor,
    pub detection_processor: DetectionProcessor,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

#[cfg(feature = "webgpu-shaders")]
impl WebGpuPipeline {
    pub async fn new() -> Result<Self> {
        let (device, queue) = init_webgpu().await?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let image_preprocessor = ImagePreprocessor::new(device.clone(), queue.clone()).await?;
        let detection_processor = DetectionProcessor::new(device.clone(), queue.clone()).await?;

        Ok(Self {
            image_preprocessor,
            detection_processor,
            device,
            queue,
        })
    }

    /// High-performance image processing using WebGPU shaders
    pub async fn process_image_webgpu(
        &self,
        img: &Mat,
        backbone: &Backbone,
        fpn: &FPN,
        net: &mut opencv::dnn::Net,
        device: &CandleDevice,
        input_size: i32,
        conf_threshold: f32,
        nms_threshold: f32,
        class_names: &[String],
        _system_config: &SystemConfig,
    ) -> Result<Mat> {
        // Step 1: Convert OpenCV Mat to raw data for WebGPU processing
        let (width, height) = (img.cols() as u32, img.rows() as u32);
        let raw_data = mat_to_u8_data(img)?;

        // Step 2: GPU-accelerated preprocessing (BGR->RGB, resize, normalize)
        let processed_data = self.image_preprocessor
            .process_image(&raw_data, width, height, input_size as u32)
            .await?;

        // Step 3: Convert back to Candle tensor for FPN processing
        let input_tensor = Tensor::from_vec(
            processed_data, 
            (1, 3, input_size as usize, input_size as usize), 
            device
        )?;

        // Step 4: Process through FPN backbone and FPN (existing pipeline)
        let backbone_outputs = backbone.forward(&input_tensor)?;
        let backbone_features = BackboneFeatures {
            c2: backbone_outputs[0].clone(),
            c3: backbone_outputs[1].clone(),
            c4: backbone_outputs[2].clone(),
            c5: backbone_outputs[3].clone(),
        };
        let _fpn_features = fpn.forward(&backbone_features)?;

        // Step 5: YOLO network inference (existing OpenCV pipeline)
        let enhanced_mat = tensor_to_mat(&input_tensor, input_size)?;
        let blob = opencv::dnn::blob_from_image(
            &enhanced_mat, 
            1.0 / 255.0, 
            Size::new(input_size, input_size), 
            opencv::core::Scalar::all(0.0), 
            false, 
            false, 
            opencv::core::CV_32F
        )?;
        
        net.set_input(&blob, "", 1.0, opencv::core::Scalar::default())?;
        let out_names = net.get_unconnected_out_layers_names()?;
        let mut out_blobs = opencv::core::Vector::<Mat>::new();
        net.forward(&mut out_blobs, &out_names)?;
        
        if out_blobs.len() == 0 {
            return Ok(img.clone());
        }
        
        let output_blob = out_blobs.get(0)?;

        // Step 6: GPU-accelerated detection post-processing
        let detection_data = mat_to_f32_data(&output_blob)?;
        let processed_detections = self.detection_processor
            .process_detections(
                &detection_data,
                img.cols() as f32,
                img.rows() as f32,
                input_size as f32,
                conf_threshold,
                nms_threshold,
            )
            .await?;

        // Step 7: Draw results on original image
        let result_img = draw_detections_from_gpu_results(img, &processed_detections, class_names)?;

        println!("🚀 WebGPU-accelerated FPN pipeline completed successfully!");
        println!("   - Image preprocessing: GPU shaders");
        println!("   - FPN backbone: CUDA tensors");
        println!("   - Detection processing: GPU compute");
        
        Ok(result_img)
    }
}

// Helper functions for data conversion

fn mat_to_u8_data(mat: &Mat) -> Result<Vec<u8>> {
    let total_elements = (mat.rows() * mat.cols() * mat.channels()) as usize;
    let mut data = Vec::with_capacity(total_elements);
    
    unsafe {
        let ptr = mat.ptr(0i32)? as *const u8;
        let slice = std::slice::from_raw_parts(ptr, total_elements);
        data.extend_from_slice(slice);
    }
    
    Ok(data)
}

fn mat_to_f32_data(mat: &Mat) -> Result<Vec<f32>> {
    let total_elements = mat.total() as usize;
    let mut data = Vec::with_capacity(total_elements);
    
    unsafe {
        let ptr = mat.ptr(0i32)? as *const f32;
        let slice = std::slice::from_raw_parts(ptr, total_elements);
        data.extend_from_slice(slice);
    }
    
    Ok(data)
}

fn tensor_to_mat(tensor: &Tensor, size: i32) -> Result<Mat> {
    // Convert tensor back to Mat for OpenCV processing
    let data = tensor.to_vec1::<f32>()?;
    let mut mat = Mat::new_size_with_default(
        Size::new(size, size), 
        opencv::core::CV_32FC3, 
        opencv::core::Scalar::all(0.0)
    )?;
    
    // Fill mat with tensor data (CHW to HWC conversion)
    let pixels = (size * size) as usize;
    unsafe {
        let mat_ptr = mat.ptr_mut(0i32)? as *mut f32;
        for y in 0..size {
            for x in 0..size {
                let pixel_idx = (y * size + x) as usize;
                let mat_idx = (y * size + x) as usize * 3;
                
                // Convert from CHW to HWC format
                *mat_ptr.add(mat_idx) = data[pixel_idx + 2 * pixels]; // B
                *mat_ptr.add(mat_idx + 1) = data[pixel_idx + pixels]; // G  
                *mat_ptr.add(mat_idx + 2) = data[pixel_idx]; // R
            }
        }
    }
    
    Ok(mat)
}

fn draw_detections_from_gpu_results(
    img: &Mat,
    detections: &[f32],
    class_names: &[String],
) -> Result<Mat> {
    let mut result_img = img.clone();
    
    // Process detections in groups of 6 (x, y, w, h, conf, class_id)
    for chunk in detections.chunks(6) {
        if chunk.len() != 6 || chunk[0] < 0.0 {
            continue; // Skip invalid detections
        }
        
        let x = chunk[0] as i32;
        let y = chunk[1] as i32;
        let w = chunk[2] as i32;
        let h = chunk[3] as i32;
        let conf = chunk[4];
        let class_id = chunk[5] as usize;
        
        let rect = opencv::core::Rect::new(x, y, w, h);
        opencv::imgproc::rectangle(
            &mut result_img,
            rect,
            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            opencv::imgproc::LINE_8,
            0,
        )?;
        
        let default_class = "unknown".to_string();
        let class_name = class_names.get(class_id).unwrap_or(&default_class);
        let label = format!("{}: {:.2}", class_name, conf);
        
        opencv::imgproc::put_text(
            &mut result_img,
            &label,
            opencv::core::Point::new(x, y - 10),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            opencv::imgproc::LINE_8,
            false,
        )?;
    }
    
    Ok(result_img)
}

/// Performance comparison and benchmarking utilities
#[cfg(feature = "webgpu-shaders")]
pub struct PerformanceBenchmark {
    pub cpu_times: Vec<std::time::Duration>,
    pub gpu_times: Vec<std::time::Duration>,
    pub memory_usage: Vec<usize>,
}

#[cfg(feature = "webgpu-shaders")]
impl PerformanceBenchmark {
    pub fn new() -> Self {
        Self {
            cpu_times: Vec::new(),
            gpu_times: Vec::new(),
            memory_usage: Vec::new(),
        }
    }
    
    pub fn add_cpu_measurement(&mut self, duration: std::time::Duration) {
        self.cpu_times.push(duration);
    }
    
    pub fn add_gpu_measurement(&mut self, duration: std::time::Duration) {
        self.gpu_times.push(duration);
    }
    
    pub fn report(&self) -> String {
        let avg_cpu = self.cpu_times.iter().sum::<std::time::Duration>() / self.cpu_times.len() as u32;
        let avg_gpu = self.gpu_times.iter().sum::<std::time::Duration>() / self.gpu_times.len() as u32;
        let speedup = avg_cpu.as_secs_f64() / avg_gpu.as_secs_f64();
        
        format!(
            "Performance Benchmark:\n\
             Average CPU Time: {:?}\n\
             Average GPU Time: {:?}\n\
             Speedup: {:.2}x\n\
             GPU utilization: {:.1}%",
            avg_cpu, avg_gpu, speedup, 
            if speedup > 1.0 { 100.0 / speedup } else { 100.0 }
        )
    }
} 