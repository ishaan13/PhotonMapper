-------------------------------------------------------------------------------
Accelerated Stochastic Progressive Photon Mapping On GPU
-------------------------------------------------------------------------------
Ishaan Singh, Yingting Xiao, Xiaoyan Zhu
-------------------------------------------------------------------------------
CIS565 FINAL PROJECT
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

Reflective bunny and refractive dragon rendered with photon mapping

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/master/renders/video/finalCover.png)

-------------------------------------------------------------------------------
Features:
-------------------------------------------------------------------------------

• Stochastic progressive photon mapping

In our photon mapping, we render the scene with direct illumination from ray tracing and indirect illumination from photon map. Photons are emitted from random points on the light source. Each photon is bounced around in a similar manner to path tracing and stored if it hits a diffuse surface. Since the photon map should only contain indirect illumination, we only store photons after their first bounce. In rendering, we trace each ray until it hits a diffuse surface or reaches the maximum number of bounces.

We need a large number of photons to produce a satisfying image. Due to limited memory on GPU, we chose to approach this in a progressive manner based on this paper: [Progressive Photon Mapping](http://graphics.ucsd.edu/~henrik/papers/progressive_photon_mapping/progressive_photon_mapping.pdf) by Toshiya Hachisuka et. al. We shoot a small number of photons every iteration and accumulate the color of the rendered image. In the ray tracing step, we jitter the sample point in pixels and the camera position such that we get nice anti-aliasing and depth of field.

• Spatial hash grid on GPU

In photon gathering, we need to find the k-nearest photons for each intersection point. Brute force search is embarrassingly slow, so we store the photons in a hash grid spatial data structure, as described in [Martin Fleisz's master thesis](http://www.inf.ed.ac.uk/publications/thesis/online/IM090675.pdf). In photon gathering, we just loop through the photons in the 9 grid cells around the intersection point. As a result, we see a dramatic performance improvement: photon gathering with 100,000 photons bouncing 5 times now takes 6 seconds instead of 3 minutes.

• Stackless KD tree on GPU

In order to render large meshes such as Stanford bunny and dragon, we use a KD tree on GPU to accelerate ray and photon tracing. Our KD tree construction and traversal is based on [Stackless KD-Tree Traversal for High Performance GPU Ray Tracing](http://www.mpi-inf.mpg.de/~guenther/StacklessGPURT/StacklessGPURT.pdf) by Stefan Popov et. al. We first construct a basic KD tree on CPU, then add ropes (pointers to a node's neighbors) to the leaf nodes. Then we convert the KD tree to GPU compatible format by storing tree nodes in an array and replacing pointers with indices in the array. In intersection detection, we traverse the tree until we hit a leaf node and test intersections with the primitives inside the node. If any primitive is intersected, we stop the traversal and return the intersection. Otherwise, we use the ropes of that leaf node to test intersection of the ray with the its neighboring nodes.

-------------------------------------------------------------------------------
User Interactions:
-------------------------------------------------------------------------------

In our program, we allow users to easily switch rendering modes with keyboard input.

1 - Direct illumination

2 - Path tracing with both direct and indirect illumination

3 - Photon map visualization

4 - Indirect illumination from photon map

5 - Direct illumination and indirect illumination

K - KD tree switch (for performance analysis)

Here are images of the same scene rendered with path tracing and photon mapping after the same number of iterations:

Photon mapping:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/master/renders/video/ppm_tetra.png)

Path tracing:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/master/renders/video/pathtrace_tetra.png)

Photon map visualization (with 500M photons):

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/master/renders/video/SphereGlow_500M.png)

-------------------------------------------------------------------------------
Additional Rendered Images and Screen Captures:
-------------------------------------------------------------------------------

Glass Stanford Dragon in a Cornell Box:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/master/renders/video/glassDragon.png)

Z depth rendering of the Stanford Dragon

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/master/renders/video/dragon.0.png)

Video with screen captures of our program running: [here](https://vimeo.com/81835028)

-------------------------------------------------------------------------------
Performance Analysis:
-------------------------------------------------------------------------------

K-nearest photon search: brute force vs. hash grid:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/performance_chart/chart_2.png)

Speed up with KD tree:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/performance_chart/chart_3.png)

KD tree construction:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/performance_chart/chart_1.png)

Different technique comparison:

![alt tag](https://raw.github.com/ishaan13/PhotonMapper/performance_chart/chart_4.png)

Follow us on [Twitter](https://twitter.com/MMFAPhoMap). 
