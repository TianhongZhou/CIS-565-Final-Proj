@group(0) @binding(0) var lambdaIn : texture_storage_2d<r32float, read>;
@group(0) @binding(1) var lambdaOut: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var uXtex    : texture_storage_2d<r32float, read>;
@group(0) @binding(3) var uYtex    : texture_storage_2d<r32float, read>;

struct S0 { dt: f32, };
@group(1) @binding(0) var<uniform> s0: S0;
struct S1 { h: f32, };
@group(1) @binding(1) var<uniform> s1: S1;
struct S2 { gamma: f32, };
@group(1) @binding(2) var<uniform> s2: S2;
struct S3 { mode: i32, };
@group(1) @binding(3) var<uniform> s3: S3;

// helpers
fn clampI(v: i32, a: i32, b: i32) -> i32 {
  if (v < a) { return a; }
  if (v > b) { return b; }
  return v;
}

// read face-centered uX (size W+1 x H)
fn read_uX(ix: i32, iy: i32) -> f32 {
  let dims = textureDimensions(uXtex);
  let w = i32(dims.x);
  let h = i32(dims.y);
  let cx = clampI(ix, 0, w-1);
  let cy = clampI(iy, 0, h-1);
  return textureLoad(uXtex, vec2<i32>(cx, cy)).x;
}

// read face-centered uY (size W x H+1)
fn read_uY(ix: i32, iy: i32) -> f32 {
  let dims = textureDimensions(uYtex);
  let w = i32(dims.x);
  let h = i32(dims.y);
  let cx = clampI(ix, 0, w-1);
  let cy = clampI(iy, 0, h-1);
  return textureLoad(uYtex, vec2<i32>(cx, cy)).x;
}

// compute cell-centered velocity by averaging surrounding faces
// cell (i,j) center at (i+0.5, j+0.5)
// u_cell.x = 0.5*(uX(i) + uX(i+1))
// u_cell.y = 0.5*(uY(j) + uY(j+1))
fn cell_velocity(ix: i32, iy: i32) -> vec2<f32> {
  let uxL = read_uX(ix, iy);
  let uxR = read_uX(ix+1, iy);
  let uyB = read_uY(ix, iy);
  let uyT = read_uY(ix, iy+1);
  return vec2<f32>(0.5 * (uxL + uxR), 0.5 * (uyB + uyT));
}

// divergence at cell center (from face-centered u)
fn divergence(ix: i32, iy: i32, h: f32) -> f32 {
  // (uX(i+1) - uX(i))/h + (uY(j+1) - uY(j))/h
  let dux = (read_uX(ix+1, iy) - read_uX(ix, iy)) / h;
  let duy = (read_uY(ix, iy+1) - read_uY(ix, iy)) / h;
  return dux + duy;
}

// Catmull-Rom cubic (as before)
fn cubicCR(v0: f32, v1: f32, v2: f32, v3: f32, t: f32) -> f32 {
  let t2 = t*t;
  let t3 = t2*t;
  let a = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3;
  let b = v0 - 2.5*v1 + 2.0*v2 - 0.5*v3;
  let c = -0.5*v0 + 0.5*v2;
  let d = v1;
  return a*t3 + b*t2 + c*t + d;
}

// bicubic sample from lambdaIn; coord is in texel-space (cell centers)
fn bicubicSampleLambda(coord: vec2<f32>) -> f32 {
  let dims = textureDimensions(lambdaIn);
  let W = i32(dims.x);
  let H = i32(dims.y);

  // floor and fractional
  let fx = floor(coord.x);
  let fy = floor(coord.y);
  let tx = coord.x - fx;
  let ty = coord.y - fy;
  let ix = i32(fx);
  let iy = i32(fy);

  // indices clamped
  let xm1 = clamp(ix-1, 0, W-1);
  let x0  = clamp(ix  , 0, W-1);
  let xp1 = clamp(ix+1, 0, W-1);
  let xp2 = clamp(ix+2, 0, W-1);

  let ym1 = clamp(iy-1, 0, H-1);
  let y0  = clamp(iy  , 0, H-1);
  let yp1 = clamp(iy+1, 0, H-1);
  let yp2 = clamp(iy+2, 0, H-1);

  let s00 = textureLoad(lambdaIn, vec2<i32>(xm1, ym1)).x;
  let s10 = textureLoad(lambdaIn, vec2<i32>(x0 , ym1)).x;
  let s20 = textureLoad(lambdaIn, vec2<i32>(xp1, ym1)).x;
  let s30 = textureLoad(lambdaIn, vec2<i32>(xp2, ym1)).x;

  let s01 = textureLoad(lambdaIn, vec2<i32>(xm1, y0 )).x;
  let s11 = textureLoad(lambdaIn, vec2<i32>(x0 , y0 )).x;
  let s21 = textureLoad(lambdaIn, vec2<i32>(xp1, y0 )).x;
  let s31 = textureLoad(lambdaIn, vec2<i32>(xp2, y0 )).x;

  let s02 = textureLoad(lambdaIn, vec2<i32>(xm1, yp1)).x;
  let s12 = textureLoad(lambdaIn, vec2<i32>(x0 , yp1)).x;
  let s22 = textureLoad(lambdaIn, vec2<i32>(xp1, yp1)).x;
  let s32 = textureLoad(lambdaIn, vec2<i32>(xp2, yp1)).x;

  let s03 = textureLoad(lambdaIn, vec2<i32>(xm1, yp2)).x;
  let s13 = textureLoad(lambdaIn, vec2<i32>(x0 , yp2)).x;
  let s23 = textureLoad(lambdaIn, vec2<i32>(xp1, yp2)).x;
  let s33 = textureLoad(lambdaIn, vec2<i32>(xp2, yp2)).x;

  let r0 = cubicCR(s00, s10, s20, s30, tx);
  let r1 = cubicCR(s01, s11, s21, s31, tx);
  let r2 = cubicCR(s02, s12, s22, s32, tx);
  let r3 = cubicCR(s03, s13, s23, s33, tx);

  return cubicCR(r0, r1, r2, r3, ty);
}

@workgroup_size(${threadsInDiffusionBlockX}, ${threadsInDiffusionBlockY}, 1)
fn transport(@builtin(global_invocation_id) globalIdx : vec3u) {

  let size = textureDimensions(diffusionIn);
  if(globalIdx.x >= size.x || globalIdx.y >= size.y) {
    return;
  }
  
  let ix = i32(globalIdx.x);
  let iy = i32(globalIdx.y);

  // cell center position in texel space (cell-centered)
  let xc = f32(ix) + 0.5;
  let yc = f32(iy) + 0.5;

  // choose velocity (we use the uX/uY bound textures directly, averaged to cell)
  let vel = cell_velocity(ix, iy);

  // divergence computed from face-centered fields
  let div = divergence(ix, iy, s1.h);

  // compute G = min(-div, -gamma * div)
  let G = min(-div, -s2.gamma * div);

  // amplification factor
  let amp = exp(G * s0.dt);

  // backtrace using cell velocity (pos_prev in texel units)
  let back_x = xc - s0.dt * vel.x / s1.h;
  let back_y = yc - s0.dt * vel.y / s1.h;

  // sample lambdaIn at backtrace pos with bicubic (coord in texel space)
  let lambda_prev = bicubicSampleLambda(vec2<f32>(back_x, back_y));

  let outv = lambda_prev * amp;

  textureStore(lambdaOut, vec2<i32>(ix, iy), vec4<f32>(outv, 0.0, 0.0, 0.0));
}
