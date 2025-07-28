use fpn_yolo_rs::{SystemConfig, Backbone, FPN, process_image_optimized};

#[cfg(feature = "webgpu-shaders")]
use fpn_yolo_rs::WebGpuPipeline;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use opencv::core::{Mat, MatTraitConst, Size};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2RGB, INTER_LINEAR};
use opencv::dnn::{read_net_from_onnx, NetTrait, NetTraitConst, DNN_BACKEND_OPENCV, DNN_TARGET_CPU};
use opencv::dnn::{DNN_BACKEND_CUDA, DNN_TARGET_CUDA};
use opencv::videoio::{VideoCapture, CAP_ANY, VideoCaptureTraitConst, VideoCaptureProperties};
use opencv::prelude::VideoCaptureTrait;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::sync::Arc;

use tokio::sync::Semaphore;
use futures::future::join_all;

fn check_system_features() {
    println!("Checking system capabilities:");
    
    match candle_core::Device::cuda_if_available(0) {
        Ok(_) => println!("  CUDA available for FPN processing"),
        Err(e) => println!("  CUDA not available: {}", e),
    }
    
    match read_net_from_onnx("yolov8m.onnx") {
        Ok(mut test_net) => {
            println!("  YOLO model (yolov8m.onnx) found");
            let cuda_available = test_net.set_preferable_backend(DNN_BACKEND_CUDA)
                .and_then(|_| test_net.set_preferable_target(DNN_TARGET_CUDA)).is_ok();
            if cuda_available {
                println!("  YOLO using CUDA acceleration");
            } else {
                println!("  YOLO using CPU");
            }
        }
        Err(_) => println!("  YOLO model not found, will use mock detections"),
    }
}

#[derive(Parser)]
#[command(name = "fpn-detection")]
#[command(about = "FPN + YOLO object detection system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Webcam {
        #[arg(short, long, default_value_t = 0)]
        camera: i32,
        #[arg(short, long, default_value = "balanced")]
        performance: String,
        #[arg(short, long)]
        workers: Option<usize>,
    },
    Dataset {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(short, long)]
        concurrency: Option<usize>,
    },
    Image {
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

struct PerformanceConfig {
    process_every_n_frames: usize,
    display_resolution: (i32, i32),
    confidence_threshold: f32,
}

impl PerformanceConfig {
    fn from_mode(mode: &str) -> Self {
        match mode {
            "turbo" => Self {
                process_every_n_frames: 8,
                display_resolution: (640, 480),
                confidence_threshold: 0.7,
            },
            "fast" => Self {
                process_every_n_frames: 4,
                display_resolution: (640, 480),
                confidence_threshold: 0.5,
            },
            "quality" => Self {
                process_every_n_frames: 1,
                display_resolution: (1280, 720),
                confidence_threshold: 0.25,
            },
            _ => Self {
                process_every_n_frames: 2,
                display_resolution: (960, 540),
                confidence_threshold: 0.35,
            }
        }
    }
}

async fn process_dataset_async(
    input_dir: &Path,
    output_dir: &Path,
    backbone: Arc<Backbone>,
    fpn: Arc<FPN>,
    system_config: Arc<SystemConfig>,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: Arc<Vec<String>>,
    max_concurrency: usize,
) -> anyhow::Result<()> {
    tokio::fs::create_dir_all(output_dir).await?;
    
    let mut entries = tokio::fs::read_dir(input_dir).await?;
    let image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"];
    let mut image_paths = Vec::new();
    
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                if image_extensions.contains(&ext_str.to_lowercase().as_str()) {
                    image_paths.push(path);
                }
            }
        }
    }
    
    let total_images = image_paths.len();
    println!("Found {} images to process", total_images);
    
    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    let processed = Arc::new(parking_lot::Mutex::new(0usize));
    
    let tasks: Vec<_> = image_paths.into_iter().map(|input_path| {
        let output_dir = output_dir.to_path_buf();
        let backbone = backbone.clone();
        let fpn = fpn.clone();
        let system_config = system_config.clone();
        let class_names = class_names.clone();
        let semaphore = semaphore.clone();
        let processed = processed.clone();
        
        tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            
            let result = tokio::task::spawn_blocking(move || {
                let mut net = match read_net_from_onnx("yolov8m.onnx") {
                    Ok(net) => net,
                    Err(_) => return Ok(()),
                };
                net.set_preferable_backend(DNN_BACKEND_OPENCV).ok();
                net.set_preferable_target(DNN_TARGET_CPU).ok();
                
                let img = imread(input_path.to_str().unwrap(), IMREAD_COLOR)?;
                if img.empty() {
                    return Err(anyhow::anyhow!("Could not load image: {:?}", input_path));
                }
                
                let result = process_image_optimized(
                    &img,
                    &backbone,
                    &fpn,
                    &mut net,
                    &system_config.device,
                    input_size,
                    conf_threshold,
                    nms_threshold,
                    &class_names,
                    &system_config,
                )?;
                
                let output_path = output_dir.join(input_path.file_name().unwrap());
                imwrite(output_path.to_str().unwrap(), &result, &opencv::core::Vector::new())?;
                
                Ok::<_, anyhow::Error>(())
            }).await;
            
            match result {
                Ok(Ok(())) => {
                    let mut count = processed.lock();
                    *count += 1;
                    if *count % 10 == 0 {
                        println!("Processed {}/{} images", *count, total_images);
                    }
                }
                Ok(Err(e)) => eprintln!("Processing error: {}", e),
                Err(e) => eprintln!("Task error: {}", e),
            }
        })
    }).collect();
    
    join_all(tasks).await;
    
    let final_count = *processed.lock();
    println!("Dataset processing complete: {}/{} images processed", final_count, total_images);
    
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    check_system_features();

    let system_config = Arc::new(SystemConfig::detect_optimal()?);

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &system_config.device);

    let backbone = Arc::new(Backbone::new(vs.pp("backbone"))?);
    let fpn = Arc::new(FPN::new(vs.pp("fpn"))?);
    
    println!("Detection system initialized");

    let input_size = 640;
    let conf_threshold = 0.25;
    let nms_threshold = 0.45;
    
    let class_names: Arc<Vec<String>> = Arc::new(vec![
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ].iter().map(|s| s.to_string()).collect());

    match cli.command {
        Commands::Webcam { camera, performance, workers } => {
            let num_workers = workers.unwrap_or(system_config.cpu_threads.min(8));
            println!("Starting webcam mode (camera: {}, performance: {})", camera, performance);
            
            run_webcam_mode(
                camera, 
                &performance, 
                backbone, 
                fpn, 
                system_config, 
                input_size, 
                conf_threshold, 
                nms_threshold, 
                class_names,
                num_workers
            ).await?;
        }
        Commands::Dataset { input, output, concurrency } => {
            let max_concurrency = concurrency.unwrap_or(system_config.cpu_threads);
            println!("Processing dataset: {:?} -> {:?}", input, output);
            
            process_dataset_async(
                &input, 
                &output, 
                backbone, 
                fpn, 
                system_config, 
                input_size, 
                conf_threshold, 
                nms_threshold, 
                class_names,
                max_concurrency
            ).await?;
        }
        Commands::Image { input, output } => {
            let output_path = output.unwrap_or_else(|| PathBuf::from("output.jpg"));
            println!("Processing image: {:?} -> {:?}", input, output_path);
            
            let system_config_for_cleanup = system_config.clone();
            
            tokio::task::spawn_blocking(move || {
                run_image_mode(&input, &output_path, &backbone, &fpn, &system_config, input_size, conf_threshold, nms_threshold, &class_names)
            }).await??;
            
            if let Some(pool) = system_config_for_cleanup.gpu_memory_pool.as_ref() {
                pool.clear();
            }
        }
    }

    Ok(())
}

async fn run_webcam_mode(
    camera_id: i32,
    performance_mode: &str,
    backbone: Arc<Backbone>,
    fpn: Arc<FPN>,
    system_config: Arc<SystemConfig>,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: Arc<Vec<String>>,
    _num_workers: usize,
) -> anyhow::Result<()> {
    let config = PerformanceConfig::from_mode(performance_mode);
    
    let mut cap = VideoCapture::new(camera_id, CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(anyhow::anyhow!("Cannot open camera {}", camera_id));
    }

    cap.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH as i32, config.display_resolution.0 as f64)?;
    cap.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT as i32, config.display_resolution.1 as f64)?;
    cap.set(VideoCaptureProperties::CAP_PROP_FPS as i32, 30.0)?;
    cap.set(VideoCaptureProperties::CAP_PROP_BUFFERSIZE as i32, 1.0)?;

    println!("Camera initialized, processing every {}th frame", config.process_every_n_frames);
    
    let mut fps_counter = 0;
    let mut fps_timer = Instant::now();
    let mut frame_skip_counter = 0;
    let mut last_processed_frame = Mat::default();
    
    let mut net = match read_net_from_onnx("yolov8m.onnx") {
        Ok(net) => net,
        Err(_) => {
            println!("YOLO model not found, using mock detections");
            return Ok(());
        }
    };
    
    let cuda_available = system_config.use_gpu && {
        net.set_preferable_backend(DNN_BACKEND_CUDA)
            .and_then(|_| net.set_preferable_target(DNN_TARGET_CUDA))
            .is_ok()
    };

    if cuda_available {
        println!("YOLO using CUDA acceleration");
    } else {
        net.set_preferable_backend(DNN_BACKEND_OPENCV).ok();
        net.set_preferable_target(DNN_TARGET_CPU).ok();
        println!("YOLO using CPU");
    }
    
    let mut frame = Mat::default();
    let mut processing_timer = Instant::now();
    
    loop {
        cap.read(&mut frame)?;
        
        if frame.empty() {
            continue;
        }

        frame_skip_counter += 1;
        if frame_skip_counter % config.process_every_n_frames == 0 {
            processing_timer = Instant::now();
            
            let processing_result = process_image_optimized(
                &frame,
                &backbone,
                &fpn,
                &mut net,
                &system_config.device,
                input_size,
                config.confidence_threshold,
                nms_threshold,
                &class_names,
                &system_config,
            );

            match processing_result {
                Ok(result) => {
                    let processing_time = processing_timer.elapsed();
                    last_processed_frame = result.clone();
                    opencv::highgui::imshow("FPN + YOLO Detection", &result)?;
                    
                    if frame_skip_counter % (config.process_every_n_frames * 10) == 0 {
                        println!("Processing time: {:.2}ms", processing_time.as_secs_f64() * 1000.0);
                    }
                }
                Err(e) => {
                    eprintln!("Processing error: {}", e);
                    opencv::highgui::imshow("FPN + YOLO Detection", &frame)?;
                }
            }
        } else {
            if !last_processed_frame.empty() {
                opencv::highgui::imshow("FPN + YOLO Detection", &last_processed_frame)?;
            } else {
                opencv::highgui::imshow("FPN + YOLO Detection", &frame)?;
            }
        }
        
        fps_counter += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            println!("FPS: {}", fps_counter);
            fps_counter = 0;
            fps_timer = Instant::now();
        }
        
        let key = opencv::highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 {
            break;
        }
    }
    
    opencv::highgui::destroy_all_windows()?;
    println!("Webcam mode ended");
    
    Ok(())
}

fn run_image_mode(
    input_path: &Path,
    output_path: &Path,
    backbone: &Backbone,
    fpn: &FPN,
    system_config: &SystemConfig,
    input_size: i32,
    conf_threshold: f32,
    nms_threshold: f32,
    class_names: &[String],
) -> anyhow::Result<()> {
    let mut net = match read_net_from_onnx("yolov8m.onnx") {
        Ok(net) => net,
        Err(_) => {
            println!("YOLO model not found, using mock detections");
            return Ok(());
        }
    };
    net.set_preferable_backend(DNN_BACKEND_OPENCV).ok();
    net.set_preferable_target(DNN_TARGET_CPU).ok();
    
    let img = imread(input_path.to_str().unwrap(), IMREAD_COLOR)?;
    if img.empty() {
        return Err(anyhow::anyhow!("Could not load image: {:?}", input_path));
    }
    
    println!("Processing image...");
    let start_time = Instant::now();
    
    let result = process_image_optimized(
        &img,
        backbone,
        fpn,
        &mut net,
        &system_config.device,
        input_size,
        conf_threshold,
        nms_threshold,
        class_names,
        system_config,
    )?;
    
    let processing_time = start_time.elapsed();
    
    imwrite(output_path.to_str().unwrap(), &result, &opencv::core::Vector::new())?;
    
    println!("Detection complete in {:?}", processing_time);
    println!("Output saved to: {:?}", output_path);
    
    Ok(())
}
