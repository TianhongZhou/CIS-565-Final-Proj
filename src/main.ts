import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU } from './renderer';
import { NaiveRenderer } from './renderers/naive';

import { setupLoaders, Scene } from './stage/scene';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';
import { canvas } from './renderer';

await initWebGPU();
setupLoaders();

let scene = new Scene();
// await scene.loadGltf('./scenes/sponza/Sponza.gltf');
let defaultSceneLoaded = false;

const camera = new Camera();

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();

const stage = new Stage(scene, camera, stats);

await loadDefaultScene();

// NaiveRenderer
var renderer: NaiveRenderer | undefined;
let clickListener: ((ev: PointerEvent) => void) | null = null;

renderer?.stop();
renderer = new NaiveRenderer(stage, 'default');

// // DiffuseRender
// var renderer: DiffuseRenderer | undefined;

// renderer?.stop();
// renderer = new DiffuseRenderer(stage);

// Set water-surface parameters:
// worldScaleXY -> the grid spans [-a,+b] in X and Z (width/depth = 2 units)
// heightScale  -> amplitude multiplier for heights sampled from the texture
const scalex = 10;
const scalez = 10;
renderer.setHeightParams(scalex, scalez, 1);

// Initialize GPU height texture and bind groups with the first frame’s data.
//renderer.updateHeight(arr, W, H);

// ---------- Scene switching (GUI) ----------
type SceneName = 'default' | 'terrain' | 'ship' | 'click';

const sceneParams = {
    scene: 'default' as SceneName,
};

gui.add(sceneParams, 'scene', ['default', 'terrain', 'ship', 'click'])
    .name('Scene')
    .onChange(async (sceneName: SceneName) => {
        await switchScene(sceneName);
    });

async function switchScene(sceneName: SceneName) {
    // Cleanup hooks from previous scene (if any)
    cleanupSceneHooks();

    // Reset simulation state by recreating renderer (fresh resources)
    renderer?.stop();
    renderer = new NaiveRenderer(stage, sceneName);
    renderer.setHeightParams(scalex, scalez, 1);

    switch (sceneName) {
        case 'terrain':
            await loadTerrainScene();
            break;
        case 'ship':
            await loadShipScene();
            break;
        case 'click':
            await loadClickScene();
            break;
        default:
            await loadDefaultScene();
            break;
    }
}

// Below are stubs/placeholders for different scene setups.
// Fill these to load assets, attach controls, and wire interactions.
async function loadDefaultScene() {
    if (!defaultSceneLoaded) {
        await scene.loadGltf('./scenes/ship/ship_pinnace_1k.gltf');
        defaultSceneLoaded = true;
    }
    console.info('Default scene active');
}

async function loadTerrainScene() {
    // TODO: load terrain mesh + water interaction.
    // e.g., await scene.loadGltf('scenes/terrain/terrain.gltf');
    // Hook into renderer/simulator to enable terrain-water coupling.
    console.info('Terrain scene stub - implement terrain + water coupling');
}

async function loadShipScene() {
    console.info('Ship scene active - use arrow keys to move imagined ball');

    if (!renderer) return;
}

async function loadClickScene() {
    console.info('Click scene active - click to add bumps');
    renderer?.resetWaterFlat();
    const handler = (ev: PointerEvent) => {
        if (!renderer) return;
        const rect = canvas.getBoundingClientRect();
        // renderer.resetWaterFlat();
        renderer.addClickBumpFromScreen(ev.clientX, ev.clientY, rect, 4.0, 8.0);
    };
    canvas.addEventListener('pointerdown', handler);
    clickListener = handler;
}

function cleanupSceneHooks() {
    if (clickListener) {
        canvas.removeEventListener('pointerdown', clickListener);
        clickListener = null;
    }
}
