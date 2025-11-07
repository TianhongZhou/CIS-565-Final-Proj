// TODO-2: implement the light clustering compute shader

// ------------------------------------
// Calculating cluster bounds:
// ------------------------------------
// For each cluster (X, Y, Z):
//     - Calculate the screen-space bounds for this cluster in 2D (XY).
//     - Calculate the depth bounds for this cluster in Z (near and far planes).
//     - Convert these screen and depth bounds into view-space coordinates.
//     - Store the computed bounding box (AABB) for the cluster.

// ------------------------------------
// Assigning lights to clusters:
// ------------------------------------
// For each cluster:
//     - Initialize a counter for the number of lights in this cluster.

//     For each light:
//         - Check if the light intersects with the cluster’s bounding box (AABB).
//         - If it does, add the light to the cluster's light list.
//         - Stop adding lights if the maximum number of lights is reached.

//     - Store the number of lights assigned to this cluster.

@group(${bindGroup_scene}) @binding(0) var<uniform>             camera : CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read>       lightSet : LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read_write> clusterSet : ClusterSet;

fn ndcFromPixel(pix: vec2<f32>, res: vec2<f32>) -> vec2<f32> {
    let uv = pix / res;
    return uv * 2.0 - vec2<f32>(1.0, 1.0);
}

fn viewPointAtDepth(ndc_xy: vec2<f32>, depth_view: f32, invProj: mat4x4f) -> vec3<f32> {
    var v = invProj * vec4<f32>(ndc_xy, -1.0, 1.0);
    v /= v.w;
    
    let scale = depth_view / -v.z;
    return v.xyz * scale;
}

fn sphereAABBIntersect(center: vec3<f32>, radius: f32, bmin: vec3f, bmax: vec3f) -> bool {
    let qx = clamp(center.x, bmin.x, bmax.x);
    let qy = clamp(center.y, bmin.y, bmax.y);
    let qz = clamp(center.z, bmin.z, bmax.z);

    let d2 = dot(center - vec3<f32>(qx, qy, qz), center - vec3<f32>(qx, qy, qz));
    return d2 <= radius * radius;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let cx = gid.x;
    let cy = gid.y;
    let cz = gid.z;

    let XS = u32(${xSlices});
    let YS = u32(${ySlices});
    let ZS = u32(${zSlices});

    if (cx >= XS || cy >= YS || cz >= ZS) { return; }

    let clusterIndex = cz * XS * YS + cy * XS + cx;

    let res  = camera.canvasResolution;  
    let near = camera.nearPlane;
    let far  = camera.farPlane;

    let tileSize = vec2<f32>(res.x / f32(XS), res.y / f32(YS));
    let pixMin   = vec2<f32>(f32(cx), f32(cy)) * tileSize;
    let pixMax   = pixMin + tileSize;

    let ndc00 = ndcFromPixel(vec2<f32>(pixMin.x, pixMin.y), res);
    let ndc10 = ndcFromPixel(vec2<f32>(pixMax.x, pixMin.y), res);
    let ndc01 = ndcFromPixel(vec2<f32>(pixMin.x, pixMax.y), res);
    let ndc11 = ndcFromPixel(vec2<f32>(pixMax.x, pixMax.y), res);

    let nz = f32(cz)     / f32(ZS);
    let fz = f32(cz + 1) / f32(ZS);
    let zNearSlice = near * pow(far / near, nz);
    let zFarSlice  = near * pow(far / near, fz);

    let invP = camera.invProjMat;

    let p0n = viewPointAtDepth(ndc00, zNearSlice, invP);
    let p1n = viewPointAtDepth(ndc10, zNearSlice, invP);
    let p2n = viewPointAtDepth(ndc01, zNearSlice, invP);
    let p3n = viewPointAtDepth(ndc11, zNearSlice, invP);

    let p0f = viewPointAtDepth(ndc00, zFarSlice, invP);
    let p1f = viewPointAtDepth(ndc10, zFarSlice, invP);
    let p2f = viewPointAtDepth(ndc01, zFarSlice, invP);
    let p3f = viewPointAtDepth(ndc11, zFarSlice, invP);

    var bmin = min(min(min(p0n, p1n), min(p2n, p3n)), min(min(p0f, p1f), min(p2f, p3f)));
    var bmax = max(max(max(p0n, p1n), max(p2n, p3n)), max(max(p0f, p1f), max(p2f, p3f)));

    var count: u32 = 0u;
    let cap = u32(${maxLightsPerCluster});
    let R   = f32(${lightRadius});

    let V = camera.viewMat;

    let total = lightSet.numLights;
    for (var i: u32 = 0u; i < total; i = i + 1u) {
        if (count >= cap) { break; }
        let Lw = lightSet.lights[i].pos;
        let Lv4 = V * vec4<f32>(Lw, 1.0);
        let Lv = Lv4.xyz; 
        if (sphereAABBIntersect(Lv, R, bmin, bmax)) {
            clusterSet.clusters[clusterIndex].lightIndices[count] = i;
            count = count + 1u;
        }
    }

    clusterSet.clusters[clusterIndex].numLights = count;
    clusterSet.clusters[clusterIndex].minBound  = bmin;
    clusterSet.clusters[clusterIndex].maxBound  = bmax;
}