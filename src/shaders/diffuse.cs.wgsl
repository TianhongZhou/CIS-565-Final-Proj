@compute
@workgroup_size(${clusteringBoundsWorkgroupSizeX}, ${clusteringBoundsWorkgroupSizeY}, ${clusteringBoundsWorkgroupSizeZ})
fn diffuse(@builtin(global_invocation_id) globalIdx: vec3u) {
}
