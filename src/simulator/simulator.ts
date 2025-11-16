import { DiffuseCS } from "./Diffuse";

export class Simulator {
  W: number;
  H: number;
  private diffuse: DiffuseCS;
  // private diffuseHeight: DiffuseCs;
  // private diffuseFlux: DiffuseCs;

  constructor(W: number, H: number, diffuse: DiffuseCS) {
    this.W = W;
    this.H = H;
    this.diffuse = diffuse;
    // this.diffuseHeight = diffuseHeight;
    // this.diffuseFlux = diffuseFlux;
  }

  simulateBulk(dt: GLfloat) {

  }

  simulateAiry(dt: GLfloat) {

  }

  transportSurface(dt: GLfloat){

  }


  simulate(dt: GLfloat) {

    //Run this 128 times, making sure to clamp dt to 0.25
    const clampedDt = Math.min(dt, 0.25);
    for (let i = 0; i < 128; i++) {
      this.diffuse.step(clampedDt);
    }

    //this.diffuseHeight.step(dt);
    //this.diffuseFlux.step(dt);

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
