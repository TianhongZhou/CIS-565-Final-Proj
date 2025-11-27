// Global constants shared by the different passes.
const PI : f32 = 3.1415926535897932384626433832795;
const TWO_PI : f32 = 6.283185307179586476925286766559;
const MAX_FFT_SIZE : u32 = 512u;  // Pad so the simple FFT scratch buffer has room.

// Multiplies two complex numbers encoded as vec2(real, imag).
fn complexMul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Bit-reversal helper for the iterative FFT implementation.
fn bitReverse(index: u32, bits: u32) -> u32 {
    var x = index;
    var rev = 0u;
    var i = 0u;
    loop {
        if (i >= bits) {
            break;
        }
        rev = (rev << 1u) | (x & 1u);
        x = x >> 1u;
        i = i + 1u;
    }
    return rev;
}

// In-place Cooley-Tukey FFT for a single row/column.
// TODO: parallelize within workgroup instead of serial loop over samples.
fn fft(count: u32, inverse: bool, dataPtr: ptr<function, array<vec2<f32>, MAX_FFT_SIZE>>) {
    if (count <= 1u) {
        return;
    }

    var bits = 0u;
    var temp = count;
    loop {
        temp = temp >> 1u;
        bits = bits + 1u;
        if (temp <= 1u) {
            break;
        }
    }

    for (var i = 0u; i < count; i = i + 1u) {
        let j = bitReverse(i, bits);
        if (i < j) {
            let t = (*dataPtr)[i];
            (*dataPtr)[i] = (*dataPtr)[j];
            (*dataPtr)[j] = t;
        }
    }

    var len = 2u;
    let sign = select(-1.0, 1.0, inverse);
    loop {
        if (len > count) {
            break;
        }
        let half = len >> 1u;
        let angle = sign * TWO_PI / f32(len);
        for (var i = 0u; i < count; i = i + len) {
            var w = vec2<f32>(1.0, 0.0);
            let wLen = vec2<f32>(cos(angle), sin(angle));
            for (var j = 0u; j < half; j = j + 1u) {
                let even = (*dataPtr)[i + j];
                let odd = complexMul(w, (*dataPtr)[i + j + half]);
                (*dataPtr)[i + j] = even + odd;
                (*dataPtr)[i + j + half] = even - odd;
                w = complexMul(w, wLen);
            }
        }
        len = len << 1u;
    }

    if (inverse) {
        let invCount = 1.0 / f32(count);
        for (var i = 0u; i < count; i = i + 1u) {
            (*dataPtr)[i] = (*dataPtr)[i] * invCount;
        }
    }
}

// Basic grid size uniform that many passes reuse.
struct GridSize {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
};

// Packed uniforms for the row/column FFTs.
struct FFTUniforms {
    width: u32,
    height: u32,
    count: u32,
    flags: u32,
};

// Uniforms needed by the exponential integrator.
struct FluxUniforms {
    dt: f32,    
    gravity: f32, 
    domainX: f32,  
    domainY: f32, 
    depth0: f32,    // H0
    depth1: f32,    // H1
    depth2: f32,    // H2
    depth3: f32,    // H3
};

@group(0) @binding(0) var heightMinusTex: texture_2d<f32>;
@group(0) @binding(1) var heightPlusTex: texture_2d<f32>;
@group(0) @binding(2) var heightMidStorage: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> gridSize: GridSize;

// Computes h_t = (h_{t-Δt/2} + h_{t+Δt/2}) / 2.
// TODO: pipe real t±Δt/2 textures instead of reusing the same high-frequency buffer.
@compute @workgroup_size(8, 8, 1)
fn airyAverage(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= gridSize.width || gid.y >= gridSize.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let minusVal = textureLoad(heightMinusTex, coord, 0).r;
    let plusVal = textureLoad(heightPlusTex, coord, 0).r;
    let avg = 0.5 * (minusVal + plusVal);
    textureStore(heightMidStorage, coord, vec4<f32>(avg, 0.0, 0.0, 0.0));
}

@group(0) @binding(0) var realInputTex: texture_2d<f32>;
@group(0) @binding(1) var complexInputTex: texture_2d<f32>;
@group(0) @binding(2) var complexOutputTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> fftUniforms: FFTUniforms;

// Runs row-wise FFT. Each invocation owns an entire row to keep logic simple.
// Flags encode whether we are consuming real data (h/q in R32) or intermediate RGBA spectra.
@compute @workgroup_size(1, 1, 1)
fn fftRows(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= fftUniforms.height) {
        return;
    }
    var data: array<vec2<f32>, MAX_FFT_SIZE>;
    let useReal = (fftUniforms.flags & 2u) != 0u;
    for (var x = 0u; x < fftUniforms.width; x = x + 1u) {
        let coord = vec2<i32>(i32(x), i32(gid.x));
        if (useReal) {
            let sample = textureLoad(realInputTex, coord, 0).r;
            data[x] = vec2<f32>(sample, 0.0);
        } else {
            let sample = textureLoad(complexInputTex, coord, 0);
            data[x] = sample.xy;
        }
    }

    fft(fftUniforms.count, (fftUniforms.flags & 1u) != 0u, &data);

    for (var x = 0u; x < fftUniforms.width; x = x + 1u) {
        let coord = vec2<i32>(i32(x), i32(gid.x));
        let value = data[x];
        textureStore(complexOutputTex, coord, vec4<f32>(value, 0.0, 0.0));
    }
}

// Column-wise FFT counterpart.
@compute @workgroup_size(1, 1, 1)
fn fftCols(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= fftUniforms.width) {
        return;
    }
    var data: array<vec2<f32>, MAX_FFT_SIZE>;
    for (var y = 0u; y < fftUniforms.height; y = y + 1u) {
        let coord = vec2<i32>(i32(gid.x), i32(y));
        let sample = textureLoad(complexInputTex, coord, 0).xy;
        data[y] = sample;
    }

    fft(fftUniforms.count, (fftUniforms.flags & 1u) != 0u, &data);

    for (var y = 0u; y < fftUniforms.height; y = y + 1u) {
        let coord = vec2<i32>(i32(gid.x), i32(y));
        let value = data[y];
        textureStore(complexOutputTex, coord, vec4<f32>(value, 0.0, 0.0));
    }
}

@group(0) @binding(0) var heightFreqTex: texture_2d<f32>;
@group(0) @binding(1) var derivXTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var derivYTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> derivSize: GridSize;

// Convert unsigned texture coordinates into signed frequency indices.
fn signedIndex(idx: u32, size: u32) -> f32 {
    if (idx > size / 2u) {
        return f32(i32(idx) - i32(size));
    }
    return f32(idx);
}

// Evaluates ∂h/∂x and ∂h/∂y = i*k*h in frequency space.
@compute @workgroup_size(8, 8, 1)
fn airyDerivatives(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= derivSize.width || gid.y >= derivSize.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let value = textureLoad(heightFreqTex, coord, 0).xy;
    let kxFactor = TWO_PI / f32(derivSize.width);
    let kyFactor = TWO_PI / f32(derivSize.height);

    let nx = signedIndex(gid.x, derivSize.width);
    let ny = signedIndex(gid.y, derivSize.height);

    let kx = kxFactor * nx;
    let ky = kyFactor * ny;

    let hRe = value.x;
    let hIm = value.y;

    let dhdx = vec2<f32>(-kx * hIm,  kx * hRe);
    let dhdy = vec2<f32>(-ky * hIm,  ky * hRe);

    textureStore(derivXTex, coord, vec4<f32>(dhdx, 0.0, 0.0));
    textureStore(derivYTex, coord, vec4<f32>(dhdy, 0.0, 0.0));
}

@group(0) @binding(0) var fluxInputTex: texture_2d<f32>;
@group(0) @binding(1) var derivInputTex: texture_2d<f32>;
@group(0) @binding(2) var fluxOutputTex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> fluxUniforms: FluxUniforms;
@group(0) @binding(4) var<uniform> fluxSize: GridSize;
@group(0) @binding(5) var depthTex: texture_2d<f32>;

// Exponential integrator for q̃ (Algorithm 2).
@compute @workgroup_size(8, 8, 1)
fn airyUpdateFlux(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= fluxSize.width || gid.y >= fluxSize.height) {
        return;
    }

    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let qValue = textureLoad(fluxInputTex, coord, 0).xy;
    let deriv = textureLoad(derivInputTex, coord, 0).xy;

    let nx = signedIndex(gid.x, fluxSize.width);
    let ny = signedIndex(gid.y, fluxSize.height);

    let kx = TWO_PI * nx / fluxUniforms.domainX;
    let ky = TWO_PI * ny / fluxUniforms.domainY;

    let kSq = kx * kx + ky * ky;
    if (kSq < 1e-6) {
        textureStore(fluxOutputTex, coord, vec4<f32>(qValue, 0.0, 0.0));
        return;
    }

    let k = sqrt(kSq);
    var depthLocal = textureLoad(depthTex, coord, 0).r;
    depthLocal = max(depthLocal, 0.01);

    let d0 = fluxUniforms.depth0;
    let d1 = fluxUniforms.depth1;
    let d2 = fluxUniforms.depth2;
    let d3 = fluxUniforms.depth3;

    var Ha = d0;
    var Hb = d1;
    var w  = 0.0;

    if (depthLocal <= d1) {
        Ha = d0;
        Hb = d1;
        let denom = max(d1 - d0, 1e-6);
        w = clamp((depthLocal - d0) / denom, 0.0, 1.0);
    } else if (depthLocal <= d2) {
        Ha = d1;
        Hb = d2;
        let denom = max(d2 - d1, 1e-6);
        w = clamp((depthLocal - d1) / denom, 0.0, 1.0);
    } else {
        Ha = d2;
        Hb = d3;
        let denom = max(d3 - d2, 1e-6);
        w = clamp((depthLocal - d2) / denom, 0.0, 1.0);
    }

    let omegaA = sqrt(fluxUniforms.gravity * k * tanh(k * Ha));
    let omegaB = sqrt(fluxUniforms.gravity * k * tanh(k * Hb));
    let omega  = mix(omegaA, omegaB, w);

    if (omega <= 0.0) {
        textureStore(fluxOutputTex, coord, vec4<f32>(qValue, 0.0, 0.0));
        return;
    }

    let coswt = cos(omega * fluxUniforms.dt);
    let sinwt = sin(omega * fluxUniforms.dt);
    let coeff = omega / kSq;

    let dComp = coeff * deriv;
    let updated = vec2<f32>(
        coswt * qValue.x - sinwt * dComp.x,
        coswt * qValue.y - sinwt * dComp.y
    );

    textureStore(fluxOutputTex, coord, vec4<f32>(updated, 0.0, 0.0));
}

@group(0) @binding(0) var fluxComplexTex: texture_2d<f32>;
@group(0) @binding(1) var fluxRealOutTex: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> realSize: GridSize;

// Converts the complex field from IFFT back into R32 real texture.
// Only the real component is required for downstream passes, so we discard the imaginary part.
@compute @workgroup_size(8, 8, 1)
fn airyWriteReal(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= realSize.width || gid.y >= realSize.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let value = textureLoad(fluxComplexTex, coord, 0).x;
    textureStore(fluxRealOutTex, coord, vec4<f32>(value, 0.0, 0.0, 0.0));
}
