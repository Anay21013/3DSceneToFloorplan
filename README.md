# Code walk-through

The problem statement was to write a code that would help us to create floorplans from 3d images. In order to create floorplans, we need to identify various planes, i.e, for floors, walls, doors, etc. The best way to do that is to use 3d scene images that are in the form of point clouds. However, it was not known whether the input images will be of this form or not. Therefore, I wrote a code which would generate the point clouds from RGB images.

### Creating Depth Maps

In order to generate point clouds, we first needed depth maps for the image. I used the MiDAS-V3

model to do the same.

```python
# Load MiDaS model
model_type = "DPT_Large"  # MiDaS v3 - Large model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
```

### Generating 3D-Point clouds

Next we use the Depth Maps to create Point clouds which would help us in identifying the planes later on in the code.

```python
def point_cloud_from_depth_map(rgb, depth, fx, fy, cx, cy):
    h, w = depth.shape
    points = []
    for v in range(h):
        for u in range(w):
            z = depth[v, u]
            if z <= 0: continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z, rgb[v, u, 2], rgb[v, u, 1], rgb[v, u, 0]])
    return np.array(points)
```

I stored the point cloud as a numpy array.

### Generating Floorplans

Next, we generate the floorplans using Open3D.

```python
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255.0)
```

Finally, we projected the 3D points into 2D to create the floorplan.

```python
# Creating 2D floorplan image
floorplan_size = 500
floorplan = np.zeros((floorplan_size, floorplan_size), dtype=np.uint8)
scaled_points = ((projected_points - np.min(projected_points, axis=0)) / (np.max(projected_points, axis=0) - np.min(projected_points, axis=0)) * (floorplan_size - 1)).astype(int)
```
