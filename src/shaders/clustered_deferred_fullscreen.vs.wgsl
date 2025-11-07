// TODO-3: implement the Clustered Deferred fullscreen vertex shader

// This shader should be very simple as it does not need all of the information passed by the the naive vertex shader.
struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
}

@vertex
fn main(@builtin(vertex_index) vid: u32) -> VSOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );

    let pos = p[vid];
    var o : VSOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.uv  = pos * 0.5 + vec2<f32>(0.5, 0.5);
    return o;
}