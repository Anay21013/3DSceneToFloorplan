import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2

# Load MiDaS model
model_type = "DPT_Large"  # MiDaS v3 - Large model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Moving model to GPU for faster evaluation
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

img = Image.open("/path_to_the_3d_image.jpg/") #kindly change the image path while evaluating the code
img = midas_transforms(img).to(device)

# depth prediction
with torch.no_grad():
    prediction = midas(img)

# Resize depth map to original image size
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[-2:],
    mode="bicubic",
    align_corners=False,
).squeeze()

depth_map = prediction.cpu().numpy()

# plt.imshow(depth_map, cmap='inferno')
# plt.title('Estimated Depth Map')
# plt.show()

# generating point cloud from depth map

# Standard Camera intrinsic parameters (can be changed according to different inputs or can be calculated from the input image itself)
fx = 1000 
fy = 1000
cx = 320 
cy = 240

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

# Load RGB image with OpenCV
rgb_image = cv2.imread('/path_to_the_3d_image.jpg/') #change the image path while evaluating

# Generate 3D point cloud from depth map
point_cloud = point_cloud_from_depth_map(rgb_image, depth_map, fx, fy, cx, cy)

# generating floor plan from 3D point cloud

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255.0)

# Downsample the point cloud
pcd = pcd.voxel_down_sample(voxel_size=0.05)

# Extracting planes for detecting floors and walls
plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Project 3D points to 2D plane to create floorplan
points = np.asarray(pcd.points)
projected_points = points[:, [0, 2]]

# Creating 2D floorplan image
floorplan_size = 500
floorplan = np.zeros((floorplan_size, floorplan_size), dtype=np.uint8)
scaled_points = ((projected_points - np.min(projected_points, axis=0)) / (np.max(projected_points, axis=0) - np.min(projected_points, axis=0)) * (floorplan_size - 1)).astype(int)

for point in scaled_points:
    cv2.circle(floorplan, tuple(point), 1, 255, -1)

# Post-process the floorplan image
kernel = np.ones((3, 3), np.uint8)
floorplan = cv2.dilate(floorplan, kernel, iterations=1)
floorplan = cv2.erode(floorplan, kernel, iterations=1)

# Display the final floorplan
plt.imshow(floorplan, cmap='gray')
plt.title('Generated Floorplan')
plt.show()

# Save the final floorplan
cv2.imwrite('floorplan.png', floorplan)