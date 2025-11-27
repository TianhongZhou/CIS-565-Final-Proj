import { DiffuseCS } from "./Diffuse";
import { AiryWaveCS } from "./AiryWaveCS";
import { TransportCS } from "./Transport";
import { VelocityCS } from "./Velocity";
import { FlowRecombineCS } from "./FlowRecombineCS";
import { HeightRecombineCS } from "./HeightRecombineCS";
import {ShallowWater} from "./ShallowWater";

export class Simulator {
  W: number;
  H: number;
  private diffuseHeight: DiffuseCS;
  private diffuseFluxX: DiffuseCS;
  private diffuseFluxY: DiffuseCS;
  private velocityCS: VelocityCS;
  private shallowWater: ShallowWater;
  private airyCS: AiryWaveCS;
  private transportFlowRateX: TransportCS;
  private transportFlowRateY: TransportCS;
  private transportHeight: TransportCS;
  private flowRecombineX: FlowRecombineCS;
  private flowRecombineY: FlowRecombineCS;
  private heightRecombine: HeightRecombineCS;
  private firstFrame: boolean = true;

  constructor(
    W: number,
    H: number,

    diffuseHeight: DiffuseCS,
    diffuseFluxX: DiffuseCS,
    diffuseFluxY: DiffuseCS,
    velocity: VelocityCS,
    shallowWater: ShallowWater,
    airy: AiryWaveCS,
    transportFlowRateX: TransportCS,
    transportFlowRateY: TransportCS,
    transportHeight: TransportCS,
    flowRecombineX: FlowRecombineCS,
    flowRecombineY: FlowRecombineCS,
    heightRecombine: HeightRecombineCS
  ) {
    this.W = W;
    this.H = H;
    this.diffuseHeight = diffuseHeight;
    this.diffuseFluxX  = diffuseFluxX;
    this.diffuseFluxY  = diffuseFluxY;
    this.velocityCS = velocity;
    this.shallowWater = shallowWater;
    this.airyCS = airy;
    this.transportFlowRateX = transportFlowRateX;
    this.transportFlowRateY = transportFlowRateY;
    this.transportHeight = transportHeight;
    this.flowRecombineX = flowRecombineX;
    this.flowRecombineY = flowRecombineY;
    this.heightRecombine = heightRecombine;
    this.firstFrame = true;
  }

  simulateDecompose(dt: GLfloat) {
    this.diffuseHeight.step(dt);
    this.diffuseFluxX.step(dt);
    this.diffuseFluxY.step(dt);
  }

  simulateBulk(dt: GLfloat) {
    this.shallowWater.step(dt);
  }

  simulateAiry(dt: GLfloat) {
    this.airyCS.step(dt);
  }

  transportSurface(dt: GLfloat){
    this.velocityCS.step();
    this.transportFlowRateX.step(dt);
    this.transportFlowRateY.step(dt);
    this.transportHeight.step(dt);
  }


  simulate(dt: GLfloat) {

    //Run this 128 times, making sure to clamp dt to 0.25
    const clampedDt = Math.min(dt, 0.25);
    this.simulateDecompose(clampedDt);

    //For shallow water, need to initialize the previous height on first frame so that it's not empty.
    if(this.firstFrame){
        // Done in ShallowWater.ts
    }
    
    this.simulateBulk(clampedDt);
    // this.simulateAiry(clampedDt);

    // this.transportSurface(clampedDt);

    // this.flowRecombineX.step();
    // this.flowRecombineY.step();

    // this.heightRecombine.step(clampedDt);
  }
}
