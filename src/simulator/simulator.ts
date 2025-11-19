import { DiffuseCS } from "./Diffuse";
import { AiryWaveCS } from "./AiryWaveCS";

export class Simulator {
  W: number;
  H: number;
  private diffuseHeight: DiffuseCS;
  private diffuseFluxX: DiffuseCS;
  private diffuseFluxY: DiffuseCS;

  private airyCS: AiryWaveCS;

  constructor(
    W: number,
    H: number,
    diffuseHeight: DiffuseCS,
    diffuseFluxX: DiffuseCS,
    diffuseFluxY: DiffuseCS,
    airy: AiryWaveCS
  ) {
    this.W = W;
    this.H = H;
    this.diffuseHeight = diffuseHeight;
    this.diffuseFluxX  = diffuseFluxX;
    this.diffuseFluxY  = diffuseFluxY;
    this.airyCS = airy;
  }

  async simulateDecompose(dt: GLfloat) {
    this.diffuseHeight.step(dt);
    this.diffuseFluxX.step(dt);
    this.diffuseFluxY.step(dt);
  }

  simulateBulk(dt: GLfloat) {

  }

  simulateAiry(dt: GLfloat) {
    this.airyCS.step(dt);
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
