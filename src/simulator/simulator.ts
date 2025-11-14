import { DiffuseCS } from "./Diffuse";

export class Simulator {
  W: number;
  H: number;
  private diffuse: DiffuseCS;

  constructor(W: number, H: number, diffuse: DiffuseCS) {
    this.W = W;
    this.H = H;
    this.diffuse = diffuse;
  }

  simulateBulk(dt: GLfloat) {

  }

  simulateAiry(dt: GLfloat) {

  }

  transportSurface(dt: GLfloat){

  }


  simulate(dt: GLfloat) {
    this.diffuse.step(dt);

    // for (let j = 1; j < this.W - 1; ++j) {
    //   for (let i = 1; i < this.H - 1; ++i) {
    //     const idx = j * this.H + i;
    //     let target = (heightIn[idx-1] + heightIn[idx+1]
    //                         + heightIn[idx+this.H] + heightIn[idx-this.H]) / 4.0;

    //     heightOut[idx] = heightIn[idx] + (target - heightIn[idx]) * 0.05;
    //   }
    // }
  }
}
