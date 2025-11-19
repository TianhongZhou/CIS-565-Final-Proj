import { DiffuseCS } from "./Diffuse";
import { AiryWave } from "./AiryWave";

export class Simulator {
  W: number;
  H: number;
  private diffuseHeight: DiffuseCS;
  private diffuseFluxX: DiffuseCS;
  private diffuseFluxY: DiffuseCS;

  private airy: AiryWave;

  constructor(
    W: number,
    H: number,
    diffuseHeight: DiffuseCS,
    diffuseFluxX: DiffuseCS,
    diffuseFluxY: DiffuseCS,
    airy: AiryWave
  ) {
    this.W = W;
    this.H = H;
    this.diffuseHeight = diffuseHeight;
    this.diffuseFluxX  = diffuseFluxX;
    this.diffuseFluxY  = diffuseFluxY;
    this.airy = airy;
  }

  async simulateDecompose(dt: GLfloat) {
    this.diffuseHeight.step(dt);
    this.diffuseFluxX.step(dt);
    this.diffuseFluxY.step(dt);

    const hHi = await this.diffuseHeight.readHighFreqToCPU();
    // TODO: need to pass t-Δt/2,t+Δt/2 into airy
    this.airy.setHeightHalfSteps(hHi, hHi);
  }

  simulateBulk(dt: GLfloat) {

  }

  simulateAiry(dt: GLfloat) {
    this.airy.step(dt);
  }

  transportSurface(dt: GLfloat){

  }


  async simulate(dt: GLfloat) {

    //Run this 128 times, making sure to clamp dt to 0.25
    const clampedDt = Math.min(dt, 0.25);
    await this.simulateDecompose(clampedDt);

    this.simulateAiry(clampedDt);
  }
}
