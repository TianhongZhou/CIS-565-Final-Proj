struct CameraUniforms {
    viewProjMat: mat4x4<f32>,
    viewMat:  mat4x4<f32>,
    invProjMat: mat4x4<f32>,
    invViewMat: mat4x4<f32>,
    canvasResolution: vec2<f32>,
    nearPlane: f32,
    farPlane: f32    
}