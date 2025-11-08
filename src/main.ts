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

// Height map render
// Allocate a heightmap (W×H). Each texel is one float (r32float).
const W=256, H=256;
const arr = new Float32Array(W * H);

// TODO: change simulate(dt, out) to the real simulation function
// renderer.setHeightUpdater((dt, out) => {
//   simulate(dt, out);
// });

// Set water-surface parameters:
// worldScaleXY -> the grid spans [-a,+b] in X and Z (width/depth = 2 units)
// heightScale  -> amplitude multiplier for heights sampled from the texture
// baseLevel    -> lifts the whole plane in world Y (0 = centered at origin)
renderer.setHeightParams(1, 1, 1, 0);

// Initialize GPU height texture and bind groups with the first frame’s data.
renderer.updateHeight(arr, W, H);
