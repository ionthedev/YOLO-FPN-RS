@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: NormalizeParams;

struct NormalizeParams {
    width: u32,
    height: u32,
    input_channels: u32,
    output_channels: u32,
    normalize_factor: f32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    std_r: f32,
    std_g: f32,
    std_b: f32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let pixel_idx = y * params.width + x;
    let total_pixels = params.width * params.height;
    
    // Read input pixel (assuming BGRA format)
    let input_pixel = input_data[pixel_idx];
    
    // Extract channels
    let b = f32((input_pixel) & 0xFFu) * params.normalize_factor;
    let g = f32((input_pixel >> 8u) & 0xFFu) * params.normalize_factor;
    let r = f32((input_pixel >> 16u) & 0xFFu) * params.normalize_factor;
    
    // Apply normalization: (value - mean) / std
    let norm_r = (r - params.mean_r) / params.std_r;
    let norm_g = (g - params.mean_g) / params.std_g;
    let norm_b = (b - params.mean_b) / params.std_b;
    
    // Store in CHW format (channels first)
    output_data[pixel_idx] = norm_r;                    // R channel
    output_data[pixel_idx + total_pixels] = norm_g;     // G channel
    output_data[pixel_idx + 2u * total_pixels] = norm_b; // B channel
} 