@group(0) @binding(0) var<storage, read> input_image: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_image: array<f32>;
@group(0) @binding(2) var<uniform> params: ResizeParams;

struct ResizeParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;
    
    if (dst_x >= params.dst_width || dst_y >= params.dst_height) {
        return;
    }
    
    // Calculate source coordinates
    let src_x = (f32(dst_x) * f32(params.src_width)) / f32(params.dst_width);
    let src_y = (f32(dst_y) * f32(params.src_height)) / f32(params.dst_height);
    
    let x0 = u32(floor(src_x));
    let y0 = u32(floor(src_y));
    let x1 = min(x0 + 1u, params.src_width - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);
    
    let fx = src_x - f32(x0);
    let fy = src_y - f32(y0);
    
    // Process each channel
    for (var c = 0u; c < params.channels; c++) {
        let channel_offset = c * params.src_width * params.src_height;
        
        // Sample 4 corner pixels
        let val00 = input_image[channel_offset + y0 * params.src_width + x0];
        let val01 = input_image[channel_offset + y0 * params.src_width + x1];
        let val10 = input_image[channel_offset + y1 * params.src_width + x0];
        let val11 = input_image[channel_offset + y1 * params.src_width + x1];
        
        // Bilinear interpolation
        let interpolated = mix(
            mix(val00, val01, fx),
            mix(val10, val11, fx),
            fy
        );
        
        // Store result
        let dst_channel_offset = c * params.dst_width * params.dst_height;
        output_image[dst_channel_offset + dst_y * params.dst_width + dst_x] = interpolated;
    }
} 