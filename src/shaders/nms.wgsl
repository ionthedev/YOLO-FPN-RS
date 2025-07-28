@group(0) @binding(0) var<storage, read> boxes: array<f32>;
@group(0) @binding(1) var<storage, read_write> suppressed: array<u32>;
@group(0) @binding(2) var<uniform> params: NMSParams;

struct NMSParams {
    num_boxes: u32,
    iou_threshold: f32,
}

fn compute_iou(box1_idx: u32, box2_idx: u32) -> f32 {
    let x1_1 = boxes[box1_idx * 6u];
    let y1_1 = boxes[box1_idx * 6u + 1u];
    let w1 = boxes[box1_idx * 6u + 2u];
    let h1 = boxes[box1_idx * 6u + 3u];
    let x2_1 = x1_1 + w1;
    let y2_1 = y1_1 + h1;
    
    let x1_2 = boxes[box2_idx * 6u];
    let y1_2 = boxes[box2_idx * 6u + 1u];
    let w2 = boxes[box2_idx * 6u + 2u];
    let h2 = boxes[box2_idx * 6u + 3u];
    let x2_2 = x1_2 + w2;
    let y2_2 = y1_2 + h2;
    
    // Calculate intersection
    let int_x1 = max(x1_1, x1_2);
    let int_y1 = max(y1_1, y1_2);
    let int_x2 = min(x2_1, x2_2);
    let int_y2 = min(y2_1, y2_2);
    
    let int_width = max(0.0, int_x2 - int_x1);
    let int_height = max(0.0, int_y2 - int_y1);
    let intersection = int_width * int_height;
    
    // Calculate union
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let union_area = area1 + area2 - intersection;
    
    if (union_area > 0.0) {
        return intersection / union_area;
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let box_idx = global_id.x;
    
    if (box_idx >= params.num_boxes) {
        return;
    }
    
    // Skip invalid boxes (marked with -1)
    if (boxes[box_idx * 6u] < 0.0) {
        suppressed[box_idx] = 1u;
        return;
    }
    
    let current_conf = boxes[box_idx * 6u + 4u];
    let current_class = u32(boxes[box_idx * 6u + 5u]);
    
    // Check against all higher confidence boxes of the same class
    for (var other_idx = 0u; other_idx < params.num_boxes; other_idx++) {
        if (other_idx == box_idx) {
            continue;
        }
        
        // Skip invalid boxes
        if (boxes[other_idx * 6u] < 0.0) {
            continue;
        }
        
        let other_conf = boxes[other_idx * 6u + 4u];
        let other_class = u32(boxes[other_idx * 6u + 5u]);
        
        // Only suppress if same class and higher confidence
        if (current_class == other_class && other_conf > current_conf) {
            let iou = compute_iou(box_idx, other_idx);
            
            if (iou > params.iou_threshold) {
                suppressed[box_idx] = 1u;
                return;
            }
        }
    }
    
    // Box survives NMS
    suppressed[box_idx] = 0u;
} 