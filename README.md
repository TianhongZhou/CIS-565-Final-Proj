# WebGPU Flowing Shallow Waves

## Overview

### Goals and Outcomes

- Reproduce the Generalizing Shallow Water Simulation with Dispersive Surface Waves (SIGGRAPH 2023) model using WebGPU.
- Achieve real-time, large-scale water wave propagation that supports reflection, refraction, obstacle interaction, and correct shallow water border interaction.
- Incorporate interactive control features, including wave painting, dropping objects into the water, and clicking to create ripples.
- Final deliverable: a WebGPU demo showcasing an interactive scene with multiple islands and boats of many sizes, as well as aforementioned interactive controls.

### Application of GPU

- The Water Surface Wavelets model evolves a 4D amplitude field across space, frequency, and direction, while the shallow water model applies realistic interactions within shallow waters such as shores.
- Updating millions of local wave components each frame requires heavy advection and diffusion operations.
- These computations are highly parallel, making GPU acceleration essential for real-time performance.
- GPU compute enables large-scale, high-resolution wave propagation and rendering at interactive frame rates.

### Algorithm

![](img/algo.png)


## Milestone #2

[Presentation](https://docs.google.com/presentation/d/1lJLH3f-Co_1rXHLxbPfNIJFWTP1bZ9lcO_hnZY8Z0Fk/edit?slide=id.p#slide=id.p)

### Result so far

![](img/m2.gif)

![](img/m2.png)

### Progress

- Simulation (Implement the entire algorithm)
    - Decompose step
    - Bulk fluid flow
    - Airy waves
        - Rewrite GPU FFT
    - Transport surface
    - Compute result

## Milestone #1

### Presentation Link

[Presentation](https://docs.google.com/presentation/d/1f0aQoDJ7CiCaOS2odb2qIY7guG_IkVA5Ko-hc2IKO7o/edit?slide=id.p#slide=id.p)

### Progress

- Simulation
    - Implement Simulator class and method functions
    - Divided simulation into 4 steps
    - Implemented decomposition
- Rendering
    - Semi-transparent water rendering with alpha blending
    - Planar reflection on the water surface using a mirrored camera

### Reference

- Stefan Jeschke and Chris Wojtan. 2023. Generalizing Shallow Water Simulations with Dispersive Surface Waves. ACM Trans. Graph. 42, 4, Article 83 (August 2023), 12 pages. https://doi.org/10.1145/3592098
- Stefan Jeschke, Tomáš Skřivan, Matthias Müller-Fischer, Nuttapong Chentanez, Miles Macklin, and Chris Wojtan. 2018. Water surface wavelets. ACM Trans. Graph. 37, 4, Article 94 (August 2018), 13 pages. https://doi.org/10.1145/3197517.3201336

- [Web FFT](https://github.com/IQEngine/WebFFT?tab=readme-ov-file)