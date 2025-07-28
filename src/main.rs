// CLI application for FPN-YOLO object detection
use fpn_yolo_rs::{
    SystemConfig, Backbone, FPN, process_image_optimized,
};

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use opencv::core::{Mat, MatTrait, MatTraitConst, Size};
use opencv::imgcodecs::{imread, imwrite, IMREAD_COLOR};
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2RGB, INTER_LINEAR};
use opencv::dnn::{read_net_from_onnx, NetTrait, NetTraitConst, DNN_BACKEND_OPENCV, DNN_TARGET_CPU};
use opencv::videoio::{VideoCapture, CAP_ANY, VideoCaptureTraitConst, VideoCaptureProperties};
use opencv::prelude::VideoCaptureTrait;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::fs;
use std::time::{Duration, Instant};
use std::sync::Arc;

// Performance-oriented imports
use tokio::sync::Semaphore;
use futures::future::join_all;

#[derive(Parser)]
#[command(name = "fpn-yolo")]
#[command(about = "High-performance FPN-enhanced YOLO object detection")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process webcam feed with parallel processing
    Webcam {
        /// Camera index (default: 0)
        #[arg(short, long, default_value_t = 0)]
        camera: i32,
        /// Performance mode: turbo, fast, balanced, quality (default: balanced)
        #[arg(short, long, default_value = "balanced")]
        performance: String,
        /// Number of parallel workers (default: auto-detect)
        #[arg(short, long)]
        workers: Option<usize>,
    },
    /// Process all images in a directory with async I/O
    Dataset {
        /// Input directory path
        #[arg(short, long)]
        input: PathBuf,
        /// Output directory path
        #[arg(short, long)]
        output: PathBuf,
        /// Maximum concurrent processing (default: CPU count)
        #[arg(short, long)]
        concurrency: Option<usize>,
    },
    /// Process a single image
    Image {
        /// Input image path
        #[arg(short, long)]
        input: PathBuf,
        /// Output image path (optional, defaults to output.jpg)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

// Performance configuration with enhanced settings
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
            _ => Self { // "balanced" or default
                process_every_n_frames: 2,
                display_resolution: (960, 540),
                confidence_threshold: 0.35,
            }
        }
    }
}

// Async dataset processing with optimized I/O
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
    // Create output directory if it doesn't exist
    tokio::fs::create_dir_all(output_dir).await?;
    
    // Read directory entries asynchronously
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
    println!("📁 Found {} images to process", total_images);
    
    // Create semaphore to limit concurrency
    let semaphore = Arc::new(Semaphore::new(max_concurrency));
    let processed = Arc::new(parking_lot::Mutex::new(0usize));
    
    // Process images in parallel with controlled concurrency
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
            
            // Load and process image
            let result = tokio::task::spawn_blocking(move || {
                let mut net = read_net_from_onnx("yolov8m.onnx")?;
                net.set_preferable_backend(DNN_BACKEND_OPENCV)?;
                net.set_preferable_target(DNN_TARGET_CPU)?;
                
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
    
    // Wait for all tasks to complete
    join_all(tasks).await;
    
    let final_count = *processed.lock();
    println!("Dataset processing complete: {}/{} images processed", final_count, total_images);
    
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Detect and configure optimal system settings
    let system_config = Arc::new(SystemConfig::detect_optimal()?);

    // Initialize FPN components with optimal device
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &system_config.device);

    let backbone = Arc::new(Backbone::new(vs.pp("backbone"))?);
    let fpn = Arc::new(FPN::new(vs.pp("fpn"))?);
    
    println!("FPN components initialized successfully");

    let input_size = 640;
    let conf_threshold = 0.25;
    let nms_threshold = 0.45;
    
    let class_names = Arc::new(vec![
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
            println!("🎥 Starting high-performance webcam mode:");
            println!("   - Camera: {}", camera);
            println!("   - Performance: {}", performance);
            println!("   - Workers: {}", num_workers);
            
            run_webcam_mode_optimized(
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
            println!("📁 Processing dataset with async I/O:");
            println!("   - Input: {:?}", input);
            println!("   - Output: {:?}", output);
            println!("   - Max concurrency: {}", max_concurrency);
            
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
            println!("Processing single image: {:?} -> {:?}", input, output_path);
            
            // Clone system_config for the cleanup later
            let system_config_for_cleanup = system_config.clone();
            
            // Use blocking task for single image
            tokio::task::spawn_blocking(move || {
                run_image_mode_optimized(&input, &output_path, &backbone, &fpn, &system_config, input_size, conf_threshold, nms_threshold, &class_names)
            }).await??;
            
            // Cleanup GPU memory pool
            if let Some(pool) = system_config_for_cleanup.gpu_memory_pool.as_ref() {
                pool.clear();
                println!("GPU memory pool cleared");
            }
        }
    }

    Ok(())
}


async fn run_webcam_mode_optimized(
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
    
    // Setup webcam processing
    let mut cap = VideoCapture::new(camera_id, CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(anyhow::anyhow!("Cannot open camera {}", camera_id));
    }

    // Set camera properties
    cap.set(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH as i32, config.display_resolution.0 as f64)?;
    cap.set(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT as i32, config.display_resolution.1 as f64)?;
    cap.set(VideoCaptureProperties::CAP_PROP_FPS as i32, 30.0)?;
    cap.set(VideoCaptureProperties::CAP_PROP_BUFFERSIZE as i32, 1.0)?;

    println!("📹 Camera initialized successfully");
    println!("🎯 Performance mode: {}", performance_mode);
    println!("   - Processing every {}th frame", config.process_every_n_frames);
    println!("   - Confidence threshold: {}", config.confidence_threshold);
    
    let mut fps_counter = 0;
    let mut fps_timer = Instant::now();
    let mut frame_skip_counter = 0;
    let mut last_processed_frame = Mat::default();
    
    // Initialize YOLO model
    let mut net = read_net_from_onnx("yolov8m.onnx")?;
    net.set_preferable_backend(DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(DNN_TARGET_CPU)?;
    
    loop {
        let mut frame = Mat::default();
        cap.read(&mut frame)?;
        
        if frame.empty() {
            continue;
        }

        frame_skip_counter += 1;
        if frame_skip_counter % config.process_every_n_frames == 0 {
            // Process frame
            match process_image_optimized(
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
            ) {
                Ok(result) => {
                    last_processed_frame = result.clone();
                    opencv::highgui::imshow("FPN-YOLO High-Performance Detection", &result)?;
                }
                Err(e) => {
                    eprintln!("Processing error: {}", e);
                    opencv::highgui::imshow("FPN-YOLO High-Performance Detection", &frame)?;
                }
            }
        } else {
            // Show last processed frame or raw frame
            if !last_processed_frame.empty() {
                opencv::highgui::imshow("FPN-YOLO High-Performance Detection", &last_processed_frame)?;
            } else {
                opencv::highgui::imshow("FPN-YOLO High-Performance Detection", &frame)?;
            }
        }
        
        // FPS counter
        fps_counter += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            println!("FPS: {} | Mode: {}", fps_counter, performance_mode);
            fps_counter = 0;
            fps_timer = Instant::now();
        }
        
        // Check for quit key
        let key = opencv::highgui::wait_key(1)?;
        if key == 'q' as i32 || key == 27 { // 'q' or ESC
            break;
        }
    }
    
    opencv::highgui::destroy_all_windows()?;
    println!("Webcam mode ended");
    
    Ok(())
}

// Optimized single image processing
fn run_image_mode_optimized(
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
    // Load YOLO model
    let mut net = read_net_from_onnx("yolov8m.onnx")?;
    net.set_preferable_backend(DNN_BACKEND_OPENCV)?;
    net.set_preferable_target(DNN_TARGET_CPU)?;
    
    // Load image
    let img = imread(input_path.to_str().unwrap(), IMREAD_COLOR)?;
    if img.empty() {
        return Err(anyhow::anyhow!("Could not load image: {:?}", input_path));
    }
    
    println!("🔍 Processing image with optimized pipeline...");
    let start_time = Instant::now();
    
    // Process image with all optimizations
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
    
    // Save result
    imwrite(output_path.to_str().unwrap(), &result, &opencv::core::Vector::new())?;
    
    println!("Detection complete!");
    println!("  Processing time: {:?}", processing_time);
    println!("  Output saved to: {:?}", output_path);
    
    Ok(())
}
