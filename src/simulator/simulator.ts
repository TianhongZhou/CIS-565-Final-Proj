import { DiffuseCS } from "./Diffuse";
import { AiryWaveCS } from "./AiryWaveCS";
import { TransportCS } from "./Transport";
import { VelocityCS } from "./Velocity";

export class Simulator {
  W: number;
  H: number;
  private diffuseHeight: DiffuseCS;
  private diffuseFluxX: DiffuseCS;
  private diffuseFluxY: DiffuseCS;
  private velocityCS: VelocityCS;
  private airyCS: AiryWaveCS;
  private transportFlowRate: TransportCS;
  private transportHeight: TransportCS;

  constructor(
    W: number,
    H: number,

    diffuseHeight: DiffuseCS,
    diffuseFluxX: DiffuseCS,
    diffuseFluxY: DiffuseCS,
    velocity: VelocityCS,
    airy: AiryWaveCS,
    transportFlowRate: TransportCS,
    transportHeight: TransportCS,
  ) {
    this.W = W;
    this.H = H;
    this.diffuseHeight = diffuseHeight;
    this.diffuseFluxX  = diffuseFluxX;
    this.diffuseFluxY  = diffuseFluxY;
    this.velocityCS = velocity;
    this.airyCS = airy;
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
    this.airyCS.step(dt);
  }

  transportSurface(dt: GLfloat){
    this.velocityCS.step();
    this.transportFlowRate.step(dt);
    this.transportHeight.step(dt);
  }


  simulate(dt: GLfloat) {

    //Run this 128 times, making sure to clamp dt to 0.25
    const clampedDt = Math.min(dt, 0.25);
    this.simulateDecompose(clampedDt);

    this.simulateAiry(clampedDt);

    this.transportSurface(clampedDt);
  }
}
