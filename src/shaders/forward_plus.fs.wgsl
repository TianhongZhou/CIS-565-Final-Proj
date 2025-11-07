// TODO-2: implement the Forward+ fragment shader

// See naive.fs.wgsl for basic fragment shader setup; this shader should use light clusters instead of looping over all lights

// ------------------------------------
// Shading process:
// ------------------------------------
// Determine which cluster contains the current fragment.
// Retrieve the number of lights that affect the current fragment from the cluster’s data.
// Initialize a variable to accumulate the total light contribution for the fragment.
// For each light in the cluster:
//     Access the light's properties using its index.
//     Calculate the contribution of the light based on its position, the fragment’s position, and the surface normal.
//     Add the calculated contribution to the total light accumulation.
// Multiply the fragment’s diffuse color by the accumulated light contribution.
// Return the final color, ensuring that the alpha component is set appropriately (typically to 1).


@group(${bindGroup_scene}) @binding(0) var<uniform>         camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read>   lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read>   clusterSet: ClusterSet;

@group(${bindGroup_material}) @binding(0) var baseColorTex : texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var baseColorSmp : sampler;

struct FSIn {
    @location(0) posWorld: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
}

@fragment
fn main(in: FSIn) -> @location(0) vec4f {
    let diffuseCol = textureSample(baseColorTex, baseColorSmp, in.uv);
    if (diffuseCol.a < 0.5) {
        discard;
    }

    let farPlane = camera.farPlane;
    let nearPlane = camera.nearPlane;

    let xSlices = u32(${xSlices});
    let ySlices = u32(${ySlices});
    let zSlices = u32(${zSlices});

    let screenPos = camera.viewProjMat * vec4f(in.posWorld, 1.0);
    let ndcPos = screenPos.xyz / screenPos.w;
    let viewPos = camera.viewMat * vec4f(in.posWorld, 1.0);

    let z = viewPos.z;
    let xCluster = u32((ndcPos.x + 1.0) * 0.5 * f32(xSlices));
    let yCluster = u32((ndcPos.y + 1.0) * 0.5 * f32(ySlices));
    let zCluster = u32(log(abs(z) / nearPlane) / log(farPlane / nearPlane) * f32(zSlices));

    let clusterIdx = zCluster * u32(xSlices) * u32(ySlices) + yCluster * u32(xSlices) + xCluster;
    
    let count = clusterSet.clusters[clusterIdx].numLights;

    var totalLightContrib = vec3<f32>(0.0);
    let n = normalize(in.nor);

    for (var i: u32 = 0u; i < count; i = i + 1u) {
        let lightIdx = clusterSet.clusters[clusterIdx].lightIndices[i];
        let light = lightSet.lights[lightIdx];
        totalLightContrib += calculateLightContrib(light, in.posWorld, n);
    }

    let finalColor = diffuseCol.rgb * totalLightContrib;
    return vec4<f32>(finalColor, 1.0);
}