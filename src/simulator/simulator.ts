import { DiffuseCS } from "./Diffuse";
import { TransportCS } from "./Transport";

export class Simulator {
  W: number;
  H: number;
  private diffuseHeight: DiffuseCS;
  private diffuseFluxX: DiffuseCS;
  private diffuseFluxY: DiffuseCS;

  private transportFlowRate: TransportCS;
  private transportHeight: TransportCS;

  constructor(
    W: number,
    H: number,

    diffuseHeight: DiffuseCS,
    diffuseFluxX: DiffuseCS,
    diffuseFluxY: DiffuseCS,

    transportFlowRate: TransportCS,
    transportHeight: TransportCS,
  ) {
    this.W = W;
    this.H = H;
    this.diffuseHeight = diffuseHeight;
    this.diffuseFluxX  = diffuseFluxX;
    this.diffuseFluxY  = diffuseFluxY;

    // TODO: transport class needs velocity here
    this.transportFlowRate = transportFlowRate;
    this.transportHeight = transportHeight;
  }

  simulateDecompose(dt: GLfloat) {
    this.diffuseHeight.step(dt);
    this.diffuseFluxX.step(dt);
    this.diffuseFluxY.step(dt);
  }

  simulateBulk(dt: GLfloat) {

  }

  simulateAiry(dt: GLfloat) {

  }

  transportSurface(dt: GLfloat){
    // compute velocity
    this.transportFlowRate.step(dt);
    this.transportHeight.step(dt);
  }


  simulate(dt: GLfloat) {

    //Run this 128 times, making sure to clamp dt to 0.25
    const clampedDt = Math.min(dt, 0.25);
    this.simulateDecompose(clampedDt);
  }
}
