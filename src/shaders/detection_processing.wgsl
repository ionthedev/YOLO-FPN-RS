@group(0) @binding(0) var<storage, read> detections: array<f32>;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;
@group(0) @binding(2) var<uniform> params: DetectionParams;

struct DetectionParams {
    num_detections: u32,
    num_classes: u32,
    conf_threshold: f32,
    nms_threshold: f32,
    img_width: f32,
    img_height: f32,
    input_size: f32,
}

struct Detection {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    confidence: f32,
    class_id: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let detection_idx = global_id.x;
    
    if (detection_idx >= params.num_detections) {
        return;
    }
    
    // YOLO output format: [x_center, y_center, width, height, class0_conf, class1_conf, ...]
    let base_idx = detection_idx;
    
    let x_center = detections[base_idx];
    let y_center = detections[params.num_detections + base_idx];
    let width = detections[2u * params.num_detections + base_idx];
    let height = detections[3u * params.num_detections + base_idx];
    
    // Find the class with maximum confidence
    var max_conf = 0.0;
    var best_class = 0u;
    
    for (var class_idx = 0u; class_idx < params.num_classes; class_idx++) {
        let conf_idx = (4u + class_idx) * params.num_detections + base_idx;
        let class_conf = detections[conf_idx];
        
        if (class_conf > max_conf) {
            max_conf = class_conf;
            best_class = class_idx;
        }
    }
    
    // Apply confidence threshold
    if (max_conf < params.conf_threshold) {
        // Mark as invalid detection
        results[detection_idx * 6u] = -1.0; // Invalid marker
        return;
    }
    
    // Scale coordinates from model input size to original image size
    let scale_x = params.img_width / params.input_size;
    let scale_y = params.img_height / params.input_size;
    
    let scaled_x_center = x_center * scale_x;
    let scaled_y_center = y_center * scale_y;
    let scaled_width = width * scale_x;
    let scaled_height = height * scale_y;
    
    // Convert from center format to corner format
    let x_min = scaled_x_center - scaled_width * 0.5;
    let y_min = scaled_y_center - scaled_height * 0.5;
    
    // Clamp to image boundaries
    let clamped_x = clamp(x_min, 0.0, params.img_width);
    let clamped_y = clamp(y_min, 0.0, params.img_height);
    let clamped_w = clamp(scaled_width, 1.0, params.img_width - clamped_x);
    let clamped_h = clamp(scaled_height, 1.0, params.img_height - clamped_y);
    
    // Store results: [x, y, w, h, confidence, class_id]
    let result_idx = detection_idx * 6u;
    results[result_idx] = clamped_x;
    results[result_idx + 1u] = clamped_y;
    results[result_idx + 2u] = clamped_w;
    results[result_idx + 3u] = clamped_h;
    results[result_idx + 4u] = max_conf;
    results[result_idx + 5u] = f32(best_class);
} 