import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';
import { NaiveRenderer } from './renderers/naive';

import { setupLoaders, Scene } from './stage/scene';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';

await initWebGPU();
setupLoaders();

let scene = new Scene();
//await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();

const stage = new Stage(scene, camera, stats);

var renderer: NaiveRenderer | undefined;

renderer?.stop();
renderer = new NaiveRenderer(stage);

const W=256, H=256;
const arr = new Float32Array(W * H);
// renderer.setHeightUpdater((dt, out) => {
//   simulate(dt, out);
// });
renderer.setHeightParams(1, 1, 1, 0);
renderer.updateHeight(arr, W, H);
