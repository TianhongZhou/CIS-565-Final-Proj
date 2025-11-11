export class Simulator {
  W: number;
  H: number;
  c: number;
  dx: number;
  damping: number;

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;

    this.c = 1.0;
    this.dx = 1.0;
    this.damping = 0.99;
  }

  simulate(dt: GLfloat, heightIn: Float32Array, heightOut: Float32Array) {
    // const { c, dx, damping } = this;

    for (let j = 1; j < this.W - 1; ++j) {
      for (let i = 1; i < this.H - 1; ++i) {
        const idx = j * this.H + i;
        let target = (heightIn[idx-1] + heightIn[idx+1]
                            + heightIn[idx+this.H] + heightIn[idx-this.H]) / 4.0;

        heightOut[idx] = heightIn[idx] + (target - heightIn[idx]) * 0.2;
      }
    }
  }
}
