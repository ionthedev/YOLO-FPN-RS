//! Python bindings for FPN-YOLO

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyArray3;

// Import our library components
use crate::{
    SystemConfig, Backbone, FPN, process_image_optimized
};

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};

// OpenCV imports with proper traits
use opencv::{
    core::{Mat, MatTraitConst, Vec3b},
    dnn::{read_net_from_onnx, NetTrait, DNN_BACKEND_OPENCV, DNN_TARGET_CPU},
    imgcodecs::{imread, IMREAD_COLOR},
    imgproc::{cvt_color, COLOR_BGR2RGB},
};

use std::sync::{Arc, Mutex};
use num_cpus;


#[pyclass]
pub struct PyFPNYOLO {
    backbone: Option<Arc<Backbone>>,
    fpn: Option<Arc<FPN>>,
    system_config: Option<Arc<SystemConfig>>,
    net: Option<Arc<Mutex<opencv::dnn::Net>>>, 
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: Arc<Vec<String>>,
    model_path: String,
    initialized: bool,
}

#[pymethods]
impl PyFPNYOLO {
    /// Create a new FPN-YOLO detector (minimal initialization)
    #[new]
    #[pyo3(signature = (model_path="yolov8m.onnx", conf_threshold=0.25, nms_threshold=0.45, input_size=640))]
    fn new(
        model_path: &str,
        conf_threshold: f32,
        nms_threshold: f32,
        input_size: i32,
    ) -> PyResult<Self> {
        // Minimal constructor - defer heavy initialization until needed
        let class_names = Arc::new(Self::get_coco_classes());

        Ok(Self {
            backbone: None,
            fpn: None,
            system_config: None,
            net: None,
            input_size,
            conf_threshold,
            nms_threshold,
            class_names,
            model_path: model_path.to_string(),
            initialized: false,
        })
    }

    /// Initialize components when first needed
    fn ensure_initialized(&mut self) -> PyResult<()> {
        if self.initialized {
            return Ok(());
        }

        // Initialize system configuration with CPU fallback for safety
        let system_config = Arc::new(
            Self::safe_system_init()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("System config error: {}", e)))?
        );

        // Initialize FPN components with safer error handling  
        let (backbone, fpn) = Self::init_fpn_components(&system_config.device)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("FPN init error: {}", e)))?;

        // Initialize YOLO network
        let net = Self::init_yolo_network(&self.model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("YOLO network error: {}", e)))?;

        self.system_config = Some(system_config);
        self.backbone = Some(backbone);
        self.fpn = Some(fpn);
        self.net = Some(Arc::new(Mutex::new(net)));
        self.initialized = true;

        Ok(())
    }

    #[pyo3(signature = (image_path, output_path=None))]
    fn detect(&mut self, py: Python, image_path: &str, output_path: Option<&str>) -> PyResult<PyObject> {
        self.ensure_initialized()?;
        
        let (result_data, result_height, result_width) = py.allow_threads(|| {
            // Load image
            let img = imread(image_path, IMREAD_COLOR)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Image load error: {}", e)))?;
            
            if img.empty() {
                return Err(pyo3::exceptions::PyValueError::new_err("Could not load image"));
            }

            // Process image
            let backbone = self.backbone.as_ref().unwrap();
            let fpn = self.fpn.as_ref().unwrap(); 
            let system_config = self.system_config.as_ref().unwrap();
            let mut net = self.net.as_ref().unwrap().lock().unwrap();
            let result = process_image_optimized(
                &img,
                backbone,
                fpn,
                &mut *net,
                &system_config.device,
                self.input_size,
                self.conf_threshold,
                self.nms_threshold,
                &self.class_names,
                system_config,
            ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Processing error: {}", e)))?;

            // Save result if output path provided
            if let Some(output) = output_path {
                opencv::imgcodecs::imwrite(output, &result, &opencv::core::Vector::new())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Save error: {}", e)))?;
            }

            // Convert result to numpy array for Python
            let result_height = result.rows() as usize;
            let result_width = result.cols() as usize;
            
            let mut result_data = Vec::with_capacity(result_height * result_width * 3);
            for y in 0..result_height {
                for x in 0..result_width {
                    let pixel = result.at_2d::<Vec3b>(y as i32, x as i32)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Pixel access error: {}", e)))?;
                    result_data.push(pixel[2]); // R
                    result_data.push(pixel[1]); // G  
                    result_data.push(pixel[0]); // B
                }
            }

            Ok::<_, PyErr>((result_data, result_height, result_width))
        })?;

        // Create numpy array from flat data
        // Reshape the flat Vec into a 3D structure
        let mut array_data = Vec::with_capacity(result_height);
        for y in 0..result_height {
            let mut row = Vec::with_capacity(result_width);
            for x in 0..result_width {
                let idx = (y * result_width + x) * 3;
                let pixel = vec![result_data[idx], result_data[idx + 1], result_data[idx + 2]];
                row.push(pixel);
            }
            array_data.push(row);
        }
        
        let array = PyArray3::from_vec3(py, &array_data)?;
        Ok(array.into())
    }


    fn detect_batch(&mut self, py: Python, image_paths: Vec<String>) -> PyResult<PyObject> {
        let results = PyList::empty(py);
        
        for image_path in image_paths {
            let result = self.detect(py, &image_path, None)?;
            results.append(result)?;
        }

        Ok(results.into())
    }


    fn get_fpn_features(&mut self, py: Python, image_path: &str) -> PyResult<PyObject> {
        self.ensure_initialized()?;
        
        let result = py.allow_threads(|| {
            // Load and preprocess image
            let img = imread(image_path, IMREAD_COLOR)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Image load error: {}", e)))?;
            
            if img.empty() {
                return Err(pyo3::exceptions::PyValueError::new_err("Could not load image"));
            }

            let mut rgb_img = Mat::default();
            cvt_color(&img, &mut rgb_img, COLOR_BGR2RGB, 0, opencv::core::AlgorithmHint::ALGO_HINT_ACCURATE)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Color conversion error: {}", e)))?;

            // Convert to tensor and process through FPN
            let system_config = self.system_config.as_ref().unwrap();
            let backbone = self.backbone.as_ref().unwrap();
            let fpn = self.fpn.as_ref().unwrap();
            
            let input_tensor = crate::mat_to_tensor_optimized(&rgb_img, &system_config.device)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tensor conversion error: {}", e)))?;
            
            let backbone_outputs = backbone.forward(&input_tensor)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Backbone forward error: {}", e)))?;
            
            // Convert Vec<Tensor> to BackboneFeatures struct
            if backbone_outputs.len() != 4 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("Expected 4 backbone feature tensors"));
            }
            let backbone_features = crate::BackboneFeatures {
                c2: backbone_outputs[0].clone(),
                c3: backbone_outputs[1].clone(),
                c4: backbone_outputs[2].clone(),
                c5: backbone_outputs[3].clone(),
            };
            
            let _fpn_features = fpn.forward(&backbone_features)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("FPN forward error: {}", e)))?;

            // For now, return a placeholder - in a full implementation you'd convert tensors to numpy
            Ok::<_, PyErr>("FPN features extracted successfully".to_string())
        })?;
        
        Ok(result.into_pyobject(py)?.into_any().unbind())
    }


    fn configure(&mut self, conf_threshold: Option<f32>, nms_threshold: Option<f32>, input_size: Option<i32>) {
        if let Some(conf) = conf_threshold {
            self.conf_threshold = conf;
        }
        if let Some(nms) = nms_threshold {
            self.nms_threshold = nms;
        }
        if let Some(size) = input_size {
            self.input_size = size;
        }
    }


    fn get_system_info(&mut self) -> PyResult<String> {
        if !self.initialized {
            return Ok(format!(
                "FPN-YOLO System Info (Not Initialized):\n- Model Path: {}\n- Input Size: {}x{}\n- Confidence Threshold: {}\n- NMS Threshold: {}",
                self.model_path,
                self.input_size,
                self.input_size, 
                self.conf_threshold,
                self.nms_threshold
            ));
        }
        
        let system_config = self.system_config.as_ref().unwrap();
        Ok(format!(
            "FPN-YOLO System Info:\n- Device: {}\n- GPU Available: {}\n- CPU Threads: {}\n- Input Size: {}x{}\n- Confidence Threshold: {}\n- NMS Threshold: {}",
            if system_config.use_gpu { "CUDA GPU" } else { "CPU" },
            system_config.use_gpu,
            system_config.cpu_threads,
            self.input_size,
            self.input_size,
            self.conf_threshold,
            self.nms_threshold
        ))
    }
}

impl PyFPNYOLO {
    /// Safe system initialization with CPU fallback
    fn safe_system_init() -> anyhow::Result<SystemConfig> {
        // Try to create system config, but force CPU if CUDA initialization fails
        match SystemConfig::detect_optimal() {
            Ok(config) => Ok(config),
            Err(_) => {
                // Fallback to CPU-only configuration
                Ok(SystemConfig {
                    device: candle_core::Device::Cpu,
                    use_gpu: false,
                    cpu_threads: num_cpus::get().saturating_sub(1).max(1),
                    gpu_memory_pool: None,
                    parallel_streams: 4,
                    use_webgpu: false,
                    use_gpu_nms: false,
                })
            }
        }
    }

    /// Initialize FPN components with error handling
    fn init_fpn_components(device: &candle_core::Device) -> anyhow::Result<(Arc<Backbone>, Arc<FPN>)> {
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let backbone = Arc::new(Backbone::new(vs.pp("backbone"))?);
        let fpn = Arc::new(FPN::new(vs.pp("fpn"))?);

        Ok((backbone, fpn))
    }

    /// Initialize YOLO network with error handling
    fn init_yolo_network(model_path: &str) -> anyhow::Result<opencv::dnn::Net> {
        let mut net = read_net_from_onnx(model_path)?;
        net.set_preferable_backend(DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(DNN_TARGET_CPU)?;
        Ok(net)
    }

    /// Get COCO class names
    fn get_coco_classes() -> Vec<String> {
        vec![
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ].iter().map(|s| s.to_string()).collect()
    }
}


#[pymodule]
pub fn fpn_yolo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFPNYOLO>()?;
    m.add("__version__", "0.1.0")?;
    m.add("__doc__", "High-performance FPN-enhanced YOLO object detection with Rust backend")?;
    Ok(())
} 