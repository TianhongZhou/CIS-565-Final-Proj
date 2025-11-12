export class Simulator {
  W: number;
  H: number;

  constructor(W: number, H: number) {
    this.W = W;
    this.H = H;
  }

  simulateBulk(dt: GLfloat) {

  }

  simulateAiry(dt: GLfloat) {

  }

  transportSurface(dt: GLfloat){

  }


  simulate(dt: GLfloat, heightIn: Float32Array, heightOut: Float32Array) {
    for (let j = 1; j < this.W - 1; ++j) {
      for (let i = 1; i < this.H - 1; ++i) {
        const idx = j * this.H + i;
        let target = (heightIn[idx-1] + heightIn[idx+1]
                            + heightIn[idx+this.H] + heightIn[idx-this.H]) / 4.0;

        heightOut[idx] = heightIn[idx] + (target - heightIn[idx]) * 0.05;
      }
    }
  }
}
