// TODO-3: implement the Clustered Deferred fullscreen fragment shader

// Similar to the Forward+ fragment shader, but with vertex information coming from the G-buffer instead.
@group(${bindGroup_scene}) @binding(0) var<uniform>       camera    : CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet  : LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> clusterSet: ClusterSet;

@group(${bindGroup_model}) @binding(0) var gPosTex : texture_2d<f32>;
@group(${bindGroup_model}) @binding(1) var gNorTex : texture_2d<f32>;
@group(${bindGroup_model}) @binding(2) var gAlbTex : texture_2d<f32>;
@group(${bindGroup_model}) @binding(3) var gSmp    : sampler;

struct FSIn {
    @location(0) uv : vec2<f32>,
}

@fragment
fn main(in: FSIn) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);

    let posWorld = textureSample(gPosTex, gSmp, uv).xyz;
    let norWorld = normalize(textureSample(gNorTex, gSmp, uv).xyz);
    let albedo   = textureSample(gAlbTex, gSmp, uv).rgb;

    if (all(posWorld == vec3<f32>(0.0))) {
        discard;
    }

    let xSlices = u32(${xSlices});
    let ySlices = u32(${ySlices});
    let zSlices = u32(${zSlices});

    let xC = clamp(u32(uv.x * f32(xSlices)), 0u, xSlices - 1u);
    let yC = clamp(u32((1.0 - uv.y) * f32(ySlices)), 0u, ySlices - 1u);

    let viewPos = camera.viewMat * vec4<f32>(posWorld, 1.0);
    let zv = max(-viewPos.z, camera.nearPlane);
    let zLin = log(zv / camera.nearPlane) / log(camera.farPlane / camera.nearPlane);
    let zC = clamp(u32(zLin * f32(zSlices)), 0u, zSlices - 1u);

    let clusterIdx = zC * xSlices * ySlices + yC * xSlices + xC;

    let count = clusterSet.clusters[clusterIdx].numLights;
    var total = vec3<f32>(0.0);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let li = clusterSet.clusters[clusterIdx].lightIndices[i];
        let light = lightSet.lights[li];
        total += calculateLightContrib(light, posWorld, norWorld);
    }

    return vec4<f32>(albedo * total, 1.0);
}