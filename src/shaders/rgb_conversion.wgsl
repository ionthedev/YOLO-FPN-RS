@group(0) @binding(0) var<storage, read> input_image: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;
@group(0) @binding(2) var<uniform> params: ImageParams;

struct ImageParams {
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    normalize_factor: f32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.target_width || y >= params.target_height) {
        return;
    }
    
    // Calculate source coordinates for bilinear interpolation
    let src_x = (f32(x) * f32(params.width)) / f32(params.target_width);
    let src_y = (f32(y) * f32(params.height)) / f32(params.target_height);
    
    let x0 = u32(floor(src_x));
    let y0 = u32(floor(src_y));
    let x1 = min(x0 + 1u, params.width - 1u);
    let y1 = min(y0 + 1u, params.height - 1u);
    
    let fx = src_x - f32(x0);
    let fy = src_y - f32(y0);
    
    // Sample 4 pixels for bilinear interpolation
    let idx00 = y0 * params.width + x0;
    let idx01 = y0 * params.width + x1;
    let idx10 = y1 * params.width + x0;
    let idx11 = y1 * params.width + x1;
    
    // Extract BGR values and convert to RGB
    let pixel00 = input_image[idx00];
    let pixel01 = input_image[idx01];
    let pixel10 = input_image[idx10];
    let pixel11 = input_image[idx11];
    
    // Extract individual channels (assuming BGRA format)
    let b00 = f32((pixel00) & 0xFFu);
    let g00 = f32((pixel00 >> 8u) & 0xFFu);
    let r00 = f32((pixel00 >> 16u) & 0xFFu);
    
    let b01 = f32((pixel01) & 0xFFu);
    let g01 = f32((pixel01 >> 8u) & 0xFFu);
    let r01 = f32((pixel01 >> 16u) & 0xFFu);
    
    let b10 = f32((pixel10) & 0xFFu);
    let g10 = f32((pixel10 >> 8u) & 0xFFu);
    let r10 = f32((pixel10 >> 16u) & 0xFFu);
    
    let b11 = f32((pixel11) & 0xFFu);
    let g11 = f32((pixel11 >> 8u) & 0xFFu);
    let r11 = f32((pixel11 >> 16u) & 0xFFu);
    
    // Bilinear interpolation for each channel
    let r_interp = mix(mix(r00, r01, fx), mix(r10, r11, fx), fy);
    let g_interp = mix(mix(g00, g01, fx), mix(g10, g11, fx), fy);
    let b_interp = mix(mix(b00, b01, fx), mix(b10, b11, fx), fy);
    
    // Normalize to [0, 1] and store in CHW format (RGB)
    let out_idx = y * params.target_width + x;
    let total_pixels = params.target_width * params.target_height;
    
    // Store as RGB in CHW format
    output_image[out_idx] = r_interp * params.normalize_factor;                    // R channel
    output_image[out_idx + total_pixels] = g_interp * params.normalize_factor;     // G channel
    output_image[out_idx + 2u * total_pixels] = b_interp * params.normalize_factor; // B channel
} 