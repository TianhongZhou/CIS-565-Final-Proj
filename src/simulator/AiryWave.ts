import webfft from "webfft";

export class AiryWave {
    private W: number;
    private H: number;
    private N: number;

    // webfft instance (inner FFT size = W; outer size = H must also be power of 2)
    private fft: any;

    private gravity = 9.81;

    // Representative water depths used in the paper: H_i in {1 m, 4 m, 16 m, 64 m}
    private depthSamples: Float32Array;

    // Smooth water depth \bar{h}(x,y), used to choose / interpolate between the precomputed depth samples
    private smoothDepth: Float32Array;

    // High-frequency height at two half time steps:
    // hHiMinus  = h̃_{t - Δt/2}
    // hHiPlus   = h̃_{t + Δt/2}
    // Algorithm 2 uses their average to obtain h̃_t.
    private hHiMinus: Float32Array;
    private hHiPlus: Float32Array;

    // High-frequency flux in x-direction, stored in real space:
    //   qxHi = q̃_x(t, x, y)
    private qxHi: Float32Array;
    private qyHi: Float32Array;

    // --- Frequency-domain buffers ---
    private hHat: Float32Array;       // \hat{h̃_t}
    private qHatX: Float32Array;      // \hat{q̃_{x,t}}
    private qHatY: Float32Array;      // \hat{q̃_{y,t}}
    private dhdxHat: Float32Array;    // ∂\hat{h̃_t} / ∂x
    private dhdyHat: Float32Array;    // ∂\hat{h̃_t} / ∂y

    // For each depth sample H_i we store:
    //  qHatDepth[i]    : \hat{q̃_{t+Δt}^i}(kx, ky)
    //  qDepthSpatial[i]: q̃_{t+Δt}^i(x, y) in real space
    private qHatDepthX: Float32Array[];
    private qHatDepthY: Float32Array[];
    private qDepthSpatialX: Float32Array[];
    private qDepthSpatialY: Float32Array[];

    // Precomputed dispersion ω_i(k) for each depth sample and
    // each frequency (kx, ky). This depends only on |k| and H_i
    // and is reused across time steps
    // omegaTable[depthIndex][linearIndex]  (linearIndex = y * W + x)
    private omegaTable: Float32Array[];

    // Temporary buffer for inverse FFT (conjugate trick)
    private tmpComplex: Float32Array;

    // Temporary real buffer for h̃_t
    private hMid: Float32Array;

    constructor(
        W: number,
        H: number,
        smoothDepth: Float32Array,
        initialHMinus: Float32Array,
        initialHPlus: Float32Array,
        initialQxHi: Float32Array,
        initialQyHi: Float32Array
    ) {
        this.W = W;
        this.H = H;
        this.N = W * H;

        this.smoothDepth = new Float32Array(smoothDepth);
        this.hHiMinus = new Float32Array(initialHMinus);
        this.hHiPlus = new Float32Array(initialHPlus);
        this.qxHi = new Float32Array(initialQxHi);
        this.qyHi     = new Float32Array(initialQyHi);

        // webfft instance: inner size = W, outer size = H (must be power-of-two)
        this.fft = new (webfft as any)(W);

        // Depth samples as in the paper
        this.depthSamples = new Float32Array([1.0, 4.0, 16.0, 64.0]);
        const numDepths = this.depthSamples.length;

        // Allocate frequency-space buffers
        this.hHat    = new Float32Array(2 * this.N);
        this.qHatX   = new Float32Array(2 * this.N);
        this.qHatY   = new Float32Array(2 * this.N);
        this.dhdxHat = new Float32Array(2 * this.N);
        this.dhdyHat = new Float32Array(2 * this.N);

        this.qHatDepthX = [];
        this.qHatDepthY = [];
        this.qDepthSpatialX = [];
        this.qDepthSpatialY = [];
        this.omegaTable = [];

        for (let i = 0; i < numDepths; ++i) {
            this.qHatDepthX.push(new Float32Array(2 * this.N));
            this.qHatDepthY.push(new Float32Array(2 * this.N));
            this.qDepthSpatialX.push(new Float32Array(this.N));
            this.qDepthSpatialY.push(new Float32Array(this.N));
            this.omegaTable.push(new Float32Array(this.N));
        }

        this.tmpComplex = new Float32Array(2 * this.N);
        this.hMid = new Float32Array(this.N);

        // Precompute dispersion ω_i(k) for all (kx, ky) and depth samples H_i
        this.precomputeOmegaTables();
    }

    // Returns the current high-frequency flux field q̃_x in real space
    getQxHi(): Float32Array {
        return this.qxHi;
    }

    getQyHi(): Float32Array {
        return this.qyHi;
    }

    // Returns the current smooth depth field \bar{h}
    getSmoothDepth(): Float32Array {
        return this.smoothDepth;
    }

    // Update the smooth depth field \bar{h}(x,y).
    setSmoothDepth(newSmoothDepth: Float32Array): void {
        if (newSmoothDepth.length !== this.N) {
            throw new Error("setSmoothDepth: size mismatch");
        }
        this.smoothDepth.set(newSmoothDepth);
    }

    // Update the half-step high-frequency heights:
    //   hHiMinus = h̃_{t - Δt/2}
    //   hHiPlus  = h̃_{t + Δt/2}
    setHeightHalfSteps(hMinus: Float32Array, hPlus: Float32Array): void {
        if (hMinus.length !== this.N || hPlus.length !== this.N) {
            throw new Error("setHeightHalfSteps: size mismatch");
        }
        this.hHiMinus.set(hMinus);
        this.hHiPlus.set(hPlus);
    }

    // Optionally overwrite q̃_x(t, x, y) directly
    setQxHi(newQx: Float32Array): void {
        if (newQx.length !== this.N) {
            throw new Error("setQxHi: size mismatch");
        }
        this.qxHi.set(newQx);
    }

    setQyHi(newQy: Float32Array): void {
        if (newQy.length !== this.N) {
            throw new Error("setQyHi: size mismatch");
        }
        this.qyHi.set(newQy);
    }

    // 2D FFT helpers (using webfft.fft2d)

    // Real-valued 2D FFT: from realIn (length N) to complexOut (length 2*N)
    private forwardFFT2DReal(realIn: Float32Array, complexOut: Float32Array): void {
        if (realIn.length !== this.N || complexOut.length !== 2 * this.N) {
            throw new Error("forwardFFT2DReal: size mismatch");
        }

        const rows: Float32Array[] = new Array(this.H);

        // Pack real data into complexOut and build row views
        for (let y = 0; y < this.H; ++y) {
            const rowOffsetReal = y * this.W;
            const rowOffsetComplex = 2 * rowOffsetReal;
            const rowView = complexOut.subarray(rowOffsetComplex, rowOffsetComplex + 2 * this.W);

            for (let x = 0; x < this.W; ++x) {
                const idx = rowOffsetReal + x;
                const v = realIn[idx];
                const idx2 = 2 * x;
                rowView[idx2] = v;
                rowView[idx2 + 1] = 0.0; // imaginary part
            }

            rows[y] = rowView;
        }

        // Perform 2D FFT in-place over the array-of-rows view.
        // webfft.fft2d returns an array of rows (often the same objects in-place).
        const outRows = this.fft.fft2d(rows);

        // Copy back into complexOut in case fft2d returns distinct arrays.
        for (let y = 0; y < this.H; ++y) {
            const srcRow = outRows[y];
            const dstOffset = 2 * y * this.W;
            complexOut.set(srcRow, dstOffset);
        }
    }

    // 2D inverse FFT using the conjugate trick:
    //   IFFT(F) = (1 / (W * H)) * conj( FFT( conj(F) ) )
    // The input "freqIn" is a complex buffer (length = 2*N).
    // The output "realOut" is a real-valued buffer (length = N).
    private inverseFFT2D(freqIn: Float32Array, realOut: Float32Array): void {
        if (freqIn.length !== 2 * this.N || realOut.length !== this.N) {
            throw new Error("inverseFFT2D: size mismatch");
        }

        const rows: Float32Array[] = new Array(this.H);

        // Conjugate into tmpComplex and build row views
        for (let y = 0; y < this.H; ++y) {
            const rowOffsetReal = y * this.W;
            const rowOffsetComplex = 2 * rowOffsetReal;
            const inRow = freqIn.subarray(rowOffsetComplex, rowOffsetComplex + 2 * this.W);
            const outRow = this.tmpComplex.subarray(rowOffsetComplex, rowOffsetComplex + 2 * this.W);

            for (let x = 0; x < this.W; ++x) {
                const idx2 = 2 * x;
                const re = inRow[idx2];
                const im = inRow[idx2 + 1];
                outRow[idx2] = re;
                outRow[idx2 + 1] = -im; // conjugate
            }

            rows[y] = outRow;
        }

        // FFT of the conjugated data
        const outRows = this.fft.fft2d(rows);

        // IFFT(F) = conj( FFT(conj(F)) ) / (W * H)
        const invScale = 1.0 / (this.W * this.H);

        for (let y = 0; y < this.H; ++y) {
            const rowOffsetReal = y * this.W;
            const rowOffsetComplex = 2 * rowOffsetReal;
            const row = outRows[y];

            for (let x = 0; x < this.W; ++x) {
                const idx = rowOffsetReal + x;
                const idx2 = 2 * x;
                const re = row[idx2];
                // const im = row[idx2 + 1]; // imaginary part is discarded

                realOut[idx] = re * invScale;
            }
        }
    }

    // Dispersion precomputation

    // Precompute ω_i(k) for all depths H_i and all frequency indices (kx, ky).
    // We use the standard Airy dispersion:
    //   ω^2 = g * |k| * tanh(|k| * H_i)
    private precomputeOmegaTables(): void {
        const W = this.W;
        const H = this.H;
        const Lx = W; // we assume Δx = 1, so domain length Lx = W
        const Ly = H; // we assume Δy = 1, so domain length Ly = H

        const kxFactor = (2.0 * Math.PI) / Lx;
        const kyFactor = (2.0 * Math.PI) / Ly;

        const numDepths = this.depthSamples.length;

        for (let y = 0; y < H; ++y) {
            let ny = y;
            if (y > H / 2) ny = y - H;
            const ky = ny * kyFactor;

            for (let x = 0; x < W; ++x) {
                let nx = x;
                if (x > W / 2) nx = x - W;
                const kx = nx * kxFactor;

                const index = y * W + x;
                const kSq = kx * kx + ky * ky;
                const k = Math.sqrt(kSq);

                if (kSq < 1e-12) {
                    // DC mode: frequency zero => ω = 0 for all depths
                    for (let d = 0; d < numDepths; ++d) {
                        this.omegaTable[d][index] = 0.0;
                    }
                    continue;
                }

                for (let d = 0; d < numDepths; ++d) {
                    const H_i = this.depthSamples[d];
                    const kh = k * H_i;
                    const omega = Math.sqrt(this.gravity * k * Math.tanh(kh));

                    const beta = 1.0; // TODO: plug in β(k, H_i) from Appendix B if desired
                    this.omegaTable[d][index] = omega / beta;
                }
            }
        }
    }

    // Time stepping (Algorithm 2)

    // Public step: apply the Airy wave update over a time interval dt.
    step(dt: number): void {
        if (dt <= 0.0) return;
        this.stepOnce(dt);
    }

    // One Airy step with time step dt.
    // Algorithm 2 (per paper):
    //   ˆh̃_t   ← FFT( (h̃_{t−Δt/2} + h̃_{t+Δt/2}) / 2 )
    //   ˆq̃_t   ← FFT( q̃_t )
    //   q̃_{t+Δt}^i ← IFFT( cos(ω_i Δt) ˆq̃_t − sin(ω_i Δt) * (ω_i / k^2) ∂ˆh̃_t/∂x )
    //   q̃_{t+Δt}(x,y) ← interpolate q̃_{t+Δt}^i(x,y) according to \bar{h}(x,y)
    private stepOnce(dt: number): void {
        const W = this.W;
        const H = this.H;
        const N = this.N;
        const numDepths = this.depthSamples.length;

        // 1) Build h̃_t = (h̃_{t−Δt/2} + h̃_{t+Δt/2}) / 2 in real space
        for (let i = 0; i < N; ++i) {
            this.hMid[i] = 0.5 * (this.hHiMinus[i] + this.hHiPlus[i]);
        }

        // 2) FFT of h̃_t and q̃_t (2D FFT over the whole domain)
        this.forwardFFT2DReal(this.hMid, this.hHat);
        this.forwardFFT2DReal(this.qxHi, this.qHatX);
        this.forwardFFT2DReal(this.qyHi, this.qHatY);

        // 3) Compute ∂ˆh̃_t / ∂x in frequency space: i*kx * \hat{h}
        const Lx = W;
        const Ly = H;
        const kxFactor = (2.0 * Math.PI) / Lx;
        const kyFactor = (2.0 * Math.PI) / Ly;

        for (let y = 0; y < H; ++y) {
            let ny = y;
            if (y > H / 2) ny = y - H;
            const ky = ny * kyFactor;

            for (let x = 0; x < W; ++x) {
                let nx = x;
                if (x > W / 2) nx = x - W;
                const kx = nx * kxFactor;

                const index = y * W + x;
                const idx2 = 2 * index;

                const hRe = this.hHat[idx2];
                const hIm = this.hHat[idx2 + 1];

                // ∂h/∂x: i*kx*h
                const dhdxRe = -kx * hIm;
                const dhdxIm =  kx * hRe;
                this.dhdxHat[idx2]     = dhdxRe;
                this.dhdxHat[idx2 + 1] = dhdxIm;

                // ∂h/∂y: i*ky*h
                const dhdyRe = -ky * hIm;
                const dhdyIm =  ky * hRe;
                this.dhdyHat[idx2]     = dhdyRe;
                this.dhdyHat[idx2 + 1] = dhdyIm;
            }
        }

        // 4) For each depth sample H_i, compute q̂_new^i(k) using the exponential integrator
        for (let d = 0; d < numDepths; ++d) {
            const qHatDX = this.qHatDepthX[d];
            const qHatDY = this.qHatDepthY[d];
            const omegaD = this.omegaTable[d];

            for (let y = 0; y < H; ++y) {
                let ny = y;
                if (y > H / 2) ny = y - H;
                const ky = ny * kyFactor;

                for (let x = 0; x < W; ++x) {
                    let nx = x;
                    if (x > W / 2) nx = x - W;
                    const kx = nx * kxFactor;

                    const index = y * W + x;
                    const idx2  = 2 * index;

                    const kSq = kx * kx + ky * ky;
                    const omega = omegaD[index];

                    if (kSq < 1e-12 || omega <= 0.0) {
                        qHatDX[idx2]     = this.qHatX[idx2];
                        qHatDX[idx2 + 1] = this.qHatX[idx2 + 1];
                        qHatDY[idx2]     = this.qHatY[idx2];
                        qHatDY[idx2 + 1] = this.qHatY[idx2 + 1];
                        continue;
                    }

                    const kInvSq = 1.0 / kSq;
                    const coswt  = Math.cos(omega * dt);
                    const sinwt  = Math.sin(omega * dt);
                    const coeff  = omega * kInvSq;

                    // --- qx ---
                    const qxRe = this.qHatX[idx2];
                    const qxIm = this.qHatX[idx2 + 1];

                    const dhdxRe = this.dhdxHat[idx2];
                    const dhdxIm = this.dhdxHat[idx2 + 1];
                    const dXRe   = coeff * dhdxRe;
                    const dXIm   = coeff * dhdxIm;

                    const newQxRe = coswt * qxRe - sinwt * dXRe;
                    const newQxIm = coswt * qxIm - sinwt * dXIm;

                    qHatDX[idx2]     = newQxRe;
                    qHatDX[idx2 + 1] = newQxIm;

                    // --- qy ---
                    const qyRe = this.qHatY[idx2];
                    const qyIm = this.qHatY[idx2 + 1];

                    const dhdyRe = this.dhdyHat[idx2];
                    const dhdyIm = this.dhdyHat[idx2 + 1];
                    const dYRe   = coeff * dhdyRe;
                    const dYIm   = coeff * dhdyIm;

                    const newQyRe = coswt * qyRe - sinwt * dYRe;
                    const newQyIm = coswt * qyIm - sinwt * dYIm;

                    qHatDY[idx2]     = newQyRe;
                    qHatDY[idx2 + 1] = newQyIm;
                }
            }
        }

        // 5) Inverse FFT for each depth: q̂_new^i(k) -> q̃_{t+Δt}^i(x,y)
        for (let d = 0; d < numDepths; ++d) {
            this.inverseFFT2D(this.qHatDepthX[d], this.qDepthSpatialX[d]);
            this.inverseFFT2D(this.qHatDepthY[d], this.qDepthSpatialY[d]);
        }

        // 6) Interpolate q̃_{t+Δt}^i(x,y) in space according to \bar{h}(x,y)
        // For each cell: choose two closest depth samples and linearly blend.
        const depths = this.depthSamples;
        const outQx  = this.qxHi;
        const outQy  = this.qyHi;

        for (let idx = 0; idx < N; ++idx) {
            const h = this.smoothDepth[idx];

            if (h <= depths[0]) {
                outQx[idx] = this.qDepthSpatialX[0][idx];
                outQy[idx] = this.qDepthSpatialY[0][idx];
                continue;
            }
            if (h >= depths[depths.length - 1]) {
                const last = depths.length - 1;
                outQx[idx] = this.qDepthSpatialX[last][idx];
                outQy[idx] = this.qDepthSpatialY[last][idx];
                continue;
            }

            let lo = 0;
            let hi = depths.length - 1;
            for (let d = 0; d < depths.length - 1; ++d) {
                if (h >= depths[d] && h <= depths[d + 1]) {
                    lo = d;
                    hi = d + 1;
                    break;
                }
            }

            const H0 = depths[lo];
            const H1 = depths[hi];
            const t  = (h - H0) / (H1 - H0);

            const qx0 = this.qDepthSpatialX[lo][idx];
            const qx1 = this.qDepthSpatialX[hi][idx];
            const qy0 = this.qDepthSpatialY[lo][idx];
            const qy1 = this.qDepthSpatialY[hi][idx];

            outQx[idx] = (1.0 - t) * qx0 + t * qx1;
            outQy[idx] = (1.0 - t) * qy0 + t * qy1;
        }
    }
}