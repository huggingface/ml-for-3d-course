# Marching Cubes

Marching Cubes is an algorithm that converts a volumetric representation to a dense mesh.

![Torus Point Cloud](https://huggingface.co/datasets/dylanebert/ml-for-3d-course/resolve/main/torus.gif)

1. **Divide the Space into Voxels:** Split the 3D space into a grid of voxels (cubic cells). The size of each voxel determines the mesh resolution.
2. **Sample the Eight Vertex Positions:** For each voxel, sample the density at the eight vertices (corners). Determine if each vertex is inside or outside the surface based on its density.
3. **Determine the Triangle Configuration:** Each voxel has eight vertices, each with two possible states (inside or outside), yielding 256 possible configurations. Each configuration corresponds to a specific triangulation pattern.

Let's walk through these steps in more detail.

### 1. Divide the Space into Voxels

The first step is to divide the 3D space into a grid of voxels. The size of each voxel will determine the resolution of the mesh.

### 2. Sample the Eight Vertex Positions

For each voxel, the algorithm samples the density at the eight vertices. Depending on the density, each vertex is classified as either `inside` or `outside` the surface.

### 3. Determine the Triangle Configuration

Each voxel's eight vertices can be in two possible states, resulting in $2^8 = 256$ possible configurations. Each configuration corresponds to a specific triangulation pattern.

![Marching Cubes Lookup](https://huggingface.co/datasets/dylanebert/ml-for-3d-course/resolve/main/MarchingCubesCases.png)

### 4. Generate the Mesh

To generate the final mesh, the algorithm "marches" through each voxel and applies the corresponding triangle configuration, hence the name "Marching Cubes".

![Marching Cubes Head](https://huggingface.co/datasets/dylanebert/ml-for-3d-course/resolve/main/Marchingcubes-head.png)

This process produces a dense, rough mesh that approximates the surface of the volume.

## Limitations

While useful for visualization, Marching Cubes meshes are mostly unsuitable for productions like games due to several limiting factors:

1. **Polygon Count:** High polygon meshes require more computation for rendering, which can significantly impact performance in real-time applications.
2. **Edge Flow:** Poor edge flow affects how the mesh deforms when animated, resulting in undesirable artifacts like creasing and pinching.
3. **Texturing:** The dense and irregular topology complicates UV mapping and texturing, resulting in texture artifacts.

![Mesh Topology](https://huggingface.co/datasets/dylanebert/ml-for-3d-course/resolve/main/topology.jpg)

### Improvements

Techniques like [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/) address some limitations by allowing the mesh vertices to move, creating smoother surfaces. This approach is used by [InstantMesh](https://huggingface.co/spaces/TencentARC/InstantMesh), the current [leading](https://huggingface.co/spaces/dylanebert/3d-arena) open-source 3D pipeline. However, the resulting meshes remain overly dense and impractical for production.

## In Practice

Cleaning up the topology of Marching Cubes output often requires more time and effort than creating a mesh from scratch. This creates a major bottleneck for ML for 3D applications. Gaussian Splatting, as discussed earlier, offers a potential solution to this bottleneck.

However, recent work has emerged that directly addresses this bottleneck, using differentiable techniques to produce low-poly meshes with higher-quality topology. This will be covered in the next section.
