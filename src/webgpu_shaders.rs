//! WebGPU WGSL compute shaders for FPN-YOLO acceleration

use wgpu::*;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use anyhow::Result;

/// GPU-based image preprocessing pipeline
pub struct ImagePreprocessor {
    device: Arc<Device>,
    queue: Arc<Queue>,
    
    // Shaders for different operations
    rgb_conversion_pipeline: ComputePipeline,
    resize_pipeline: ComputePipeline,
    normalization_pipeline: ComputePipeline,
    
    // Buffers for image processing
    input_buffer: Buffer,
    output_buffer: Buffer,
    staging_buffer: Buffer,
}

/// Detection post-processing on GPU
pub struct DetectionProcessor {
    device: Arc<Device>,
    queue: Arc<Queue>,
    
    // Detection processing pipeline
    detection_pipeline: ComputePipeline,
    nms_pipeline: ComputePipeline,
    
    // Buffers for detections
    detection_buffer: Buffer,
    results_buffer: Buffer,
    staging_buffer: Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImageParams {
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    normalize_factor: f32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DetectionParams {
    num_detections: u32,
    num_classes: u32,
    conf_threshold: f32,
    nms_threshold: f32,
    img_width: f32,
    img_height: f32,
    input_size: f32,
    _padding: u32,
}

impl ImagePreprocessor {
    pub async fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        // RGB conversion shader
        let rgb_conversion_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("RGB Conversion Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/rgb_conversion.wgsl").into()),
        });

        // Resize shader with bilinear interpolation
        let resize_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Resize Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/resize.wgsl").into()),
        });

        // Normalization shader
        let normalization_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Normalization Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/normalize.wgsl").into()),
        });

        // Create compute pipelines
        let rgb_conversion_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("RGB Conversion Pipeline"),
            layout: None,
            module: &rgb_conversion_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let resize_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Resize Pipeline"),
            layout: None,
            module: &resize_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let normalization_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Normalization Pipeline"),
            layout: None,
            module: &normalization_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers (size will be adjusted based on actual image dimensions)
        let buffer_size = 1920 * 1080 * 4 * 4; // Conservative estimate for full HD RGBA
        
        let input_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Input Image Buffer"),
            size: buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Output Image Buffer"),
            size: buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            rgb_conversion_pipeline,
            resize_pipeline,
            normalization_pipeline,
            input_buffer,
            output_buffer,
            staging_buffer,
        })
    }

    /// Process image on GPU: BGR->RGB, resize, normalize
    pub async fn process_image(&self, input_data: &[u8], width: u32, height: u32, target_size: u32) -> Result<Vec<f32>> {
        let params = ImageParams {
            width,
            height,
            target_width: target_size,
            target_height: target_size,
            normalize_factor: 1.0 / 255.0,
            _padding: [0; 3],
        };

        // Upload image data
        self.queue.write_buffer(&self.input_buffer, 0, input_data);

        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Image Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Image Processing Bind Group"),
            layout: &self.rgb_conversion_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute shaders
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Image Processing Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Image Processing Pass"),
                timestamp_writes: None,
            });

            // RGB conversion + resize + normalization in sequence
            compute_pass.set_pipeline(&self.rgb_conversion_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((target_size + 15) / 16, (target_size + 15) / 16, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_buffer, 0, (target_size * target_size * 3 * 4) as u64);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        receiver.receive().await.unwrap()?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }
}

impl DetectionProcessor {
    pub async fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        // Detection processing shader
        let detection_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Detection Processor Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/detection_processing.wgsl").into()),
        });

        // NMS shader
        let nms_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("NMS Shader"),
            source: ShaderSource::Wgsl(include_str!("./shaders/nms.wgsl").into()),
        });

        let detection_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Detection Pipeline"),
            layout: None,
            module: &detection_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let nms_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("NMS Pipeline"),
            layout: None,
            module: &nms_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create buffers for detection data
        let max_detections = 25200; // Common YOLO output size
        let detection_buffer_size = max_detections * 84 * 4; // 84 values per detection

        let detection_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Detection Buffer"),
            size: detection_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Results Buffer"),
            size: (max_detections * 16) as u64, // x, y, w, h per detection
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Detection Staging Buffer"),
            size: (max_detections * 16) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            detection_pipeline,
            nms_pipeline,
            detection_buffer,
            results_buffer,
            staging_buffer,
        })
    }

    /// Process detections on GPU with parallel confidence filtering and coordinate transformation
    pub async fn process_detections(
        &self,
        detection_data: &[f32],
        img_width: f32,
        img_height: f32,
        input_size: f32,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<Vec<f32>> {
        let params = DetectionParams {
            num_detections: detection_data.len() as u32 / 84,
            num_classes: 80,
            conf_threshold,
            nms_threshold,
            img_width,
            img_height,
            input_size,
            _padding: 0,
        };

        // Upload detection data
        self.queue.write_buffer(&self.detection_buffer, 0, bytemuck::cast_slice(detection_data));

        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Detection Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Detection Processing Bind Group"),
            layout: &self.detection_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.detection_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.results_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute detection processing
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Detection Processing Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Detection Processing Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.detection_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with optimal workgroup size for detection processing
            let workgroups = (params.num_detections + 255) / 256;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy results back
        encoder.copy_buffer_to_buffer(&self.results_buffer, 0, &self.staging_buffer, 0, self.staging_buffer.size());
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        receiver.receive().await.unwrap()?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }
}

/// Initialize WebGPU for compute operations
pub async fn init_webgpu() -> Result<(Device, Queue)> {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to request WebGPU adapter: {}", e))?;

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await?;

    Ok((device, queue))
} 