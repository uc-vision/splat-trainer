from dataclasses import dataclass, replace
from functools import cached_property
import math
import time
from typing import Optional, Tuple
from beartype import beartype
from beartype.typing import List
import cv2
import torch
import trimesh

import numpy as np
import pyrender

from splat_trainer.camera_table import camera_table
from splat_trainer.util.misc import lerp
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.visibility.query_points import point_visibility



def to_matrix(intrinsics:np.ndarray) -> np.ndarray:
  fx, fy, cx, cy = intrinsics.T
  m = np.eye(3, dtype=intrinsics.dtype)
  m = m.reshape(1, 3, 3).repeat(intrinsics.shape[0], axis=0)
  m[..., 0, 0] = fx
  m[..., 1, 1] = fy
  m[..., 0, 2] = cx
  m[..., 1, 2] = cy
  return m


def split_rt(
    transform: np.ndarray,  # (batch_size, 4, 4)
) -> Tuple[np.ndarray, np.ndarray]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    return R.contiguous(), t.contiguous()

def join_rt(r, t):
  assert r.shape[-2:] == (3, 3), f"Expected (..., 3, 3) tensor, got: {r.shape}"
  assert t.shape[-1] == 3, f"Expected (..., 3) tensor, got: {t.shape}"

  prefix = t.shape[:-1]
  assert prefix == t.shape[:-1], f"Expected same prefix shape, got: {r.shape} {t.shape}"

  T = np.tile(np.eye(4, dtype=r.dtype).reshape((1,) * len(prefix) + (4, 4)), prefix + (1, 1))

  T[..., 0:3, 0:3] = r
  T[..., 0:3, 3] = t
  return T

@beartype
@dataclass(kw_only=True)
class Camera:
  intrinsics: np.ndarray
  camera_t_world: np.ndarray
  image_size: Tuple[int, int]
  depth_range: Tuple[float, float]
  camera_idx: int
  frame_idx: int
  label: camera_table.Label

  @property
  def rotation(self) -> np.ndarray:
    return self.world_t_camera[:3, :3]
  
  @property
  def position(self) -> np.ndarray:
    return self.world_t_camera[:3, 3]

  @cached_property
  def world_t_camera(self) -> np.ndarray:
    return np.linalg.inv(self.camera_t_world)

  @property
  def fov(self) -> Tuple[float, float]:
    return tuple(2.0 * math.atan(0.5 * size / f) 
                 for f, size in zip(self.intrinsics[:2], self.image_size))
  
  @property
  def intrinsic_matrix(self) -> np.ndarray:
    return to_matrix(self.intrinsics)
  
  @property
  def forward(self) -> np.ndarray:
    return self.world_t_camera[:3, 2]
  
  @property
  def right(self) -> np.ndarray:
    return self.world_t_camera[:3, 0]
  
  @property
  def up(self) -> np.ndarray:
    return self.world_t_camera[:3, 1]
  
  def translate(self, dir:np.ndarray) -> 'Camera':
    dir = self.world_t_camera @ np.concatenate([dir, np.zeros(1)], axis=-1)
    pos = self.position + dir[:3]
    return replace(self, camera_t_world=np.linalg.inv(join_rt(self.rotation, pos)))
  
  @property
  def focal_length(self) -> np.ndarray:
    return self.intrinsics[:2]
  
  @property
  def principal_point(self) -> np.ndarray:
    return self.intrinsics[2:]
  
  @property
  def aspect_ratio(self) -> float:
    return self.image_size[0] / self.image_size[1]
  
  @property
  def has_label(self, label:camera_table.Label) -> bool:
    return bool(self.label & label)
  
  @staticmethod
  def from_torch(camera:camera_table.Camera) -> 'Camera':
    return Camera(
      intrinsics=camera.intrinsics.cpu().numpy(),
      camera_t_world=camera.camera_t_world.cpu().numpy(),
      image_size=camera.image_size,
      depth_range=camera.depth_range,
      camera_idx=camera.camera_idx,
      frame_idx=camera.frame_idx,
      label=camera.label
    )


def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return np.concatenate([points, np.ones(shape, dtype=np.float32)], axis=-1)


def batch_transform(transforms, points):
  assert points.shape[
      -1] == 3 and points.ndim == 2, 'transform_points: expected 3d points of Nx3, got:' + str(
          points.shape)
  assert transforms.shape[-2:] == (
      4, 4
  ) and transforms.ndim == 3, 'transform_points: expected Mx4x4, got:' + str(
      transforms.shape)

  homog = make_homog(points)
  transformed = transforms.reshape(transforms.shape[0], 1, 4, 4) @ homog.reshape(
      1, *homog.shape, 1)

  return transformed[..., 0].reshape([transforms.shape[0], -1, 4])[:, :, :3]


def instance_meshes(mesh:trimesh.Trimesh, transforms:np.array, 
                    face_colors:Optional[np.ndarray]=None,
                    vertex_colors:Optional[np.ndarray]=None):
  n = transforms.shape[0]
  
  vertices = batch_transform(transforms, mesh.vertices)
  
  offsets = np.arange(n).reshape(n, 1, 1) * mesh.vertices.shape[0] 
  faces = mesh.faces.reshape(1, -1, 3) + offsets

  if face_colors is not None:
    face_colors = np.repeat(face_colors, mesh.faces.shape[0], axis=0).reshape(-1, 3)

  if vertex_colors is not None:
    vertex_colors = np.repeat(vertex_colors, mesh.vertices.shape[0], axis=0).reshape(-1, 3)

  mesh = trimesh.Trimesh(vertices=vertices.reshape(-1, 3), 
                         faces=faces.reshape(-1, 3),
                         face_colors=face_colors,
                         vertex_colors=vertex_colors)
  

  mesh.fix_normals()
  return mesh

def camera_marker(camera:Camera, scale=0.1):
  x, y = [0.5 * np.tan(a/2) for a in camera.fov]

  points = np.array([
    [0, 0, 0],
    [-x,  y, 1],
    [x,   y, 1],
    [x,  -y, 1],
    [-x, -y, 1]
    ])
  

  triangles = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 1],

    [1, 2, 3],
    [1, 3, 4]
  ], dtype=np.int32) 

  mesh = trimesh.Trimesh(vertices=points * scale, faces=triangles, process=False) 
  return mesh
    


def make_camera_markers(cameras:camera_table.Cameras, 
                        colors:np.ndarray | tuple[float, float, float], scale:float=0.1):
    
    if isinstance(colors, tuple):
      colors = np.array([colors]).reshape(1, 3).repeat(cameras.shape[0], axis=0)

    mesh = camera_marker(Camera.from_torch(cameras[0].item()), scale)
    markers = instance_meshes(mesh, cameras.world_t_camera.cpu().numpy(), 
                              face_colors=colors)

    markers = pyrender.Mesh.from_trimesh(
        markers, smooth=False,
    )

  
    return markers




def make_sphere(radius=1.0, subdivisions=3, color=(0.0, 0.0, 1.0)):
  sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)

  material = pyrender.MetallicRoughnessMaterial(doubleSided=True, wireframe=False, smooth=False, baseColorFactor=(*[color * 255], 255))
  return pyrender.Mesh.from_trimesh(sphere, smooth=True, material=material)

def translations(positions:np.ndarray):
  transforms = np.tile(np.eye(4, dtype=positions.dtype), (positions.shape[0], 1, 1))
  transforms[..., :3, 3] = positions
  return transforms

def make_spheres(radius=0.1, positions=np.ndarray, colors=np.ndarray):
  sphere = trimesh.creation.icosphere(radius=radius, subdivisions=3)


  spheres = instance_meshes(sphere, translations(positions), vertex_colors=colors)
  return pyrender.Mesh.from_trimesh(spheres, smooth=True)

def fov_to_focal(fov, image_size):
  return image_size / (2 * np.tan(fov / 2))

flip_yz = np.array([
  [1, 0, 0],
  [0, -1, 0],
  [0, 0, -1]
])

def normalize(v):
  return v / np.linalg.norm(v)

def to_pyrender_camera(camera:camera_table.Camera, viewport_size:Tuple[int, int]):
    camera = Camera.from_torch(camera)

    scaling = max(viewport_size) / max(camera.image_size)

    fx, fy = camera.focal_length * scaling
    cx, cy = camera.principal_point * scaling

    pr_camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
        znear=camera.depth_range[0], zfar=camera.depth_range[1])
    
    rotation = camera.rotation  @ flip_yz 
    m = join_rt(rotation, camera.position)

    return pyrender.Node(camera=pr_camera, matrix=m)


def look_at(eye, target, up=np.array([0., 0., 1.])):
  forward = normalize(target - eye)
  left = normalize(np.cross(up, forward))
  true_up = np.cross(forward, left)
  return np.stack([left, true_up, forward])


def look_at_pose(eye, target, up=np.array([0., 0., 1.])):
  pose = np.eye(4)
  pose[:3, :3] = look_at(eye, target, up)
  pose[:3, 3] = eye
  return pose

@beartype
class CameraViewer:
  def __init__(self, cameras:camera_table.Cameras, points:PointCloud, marker_size:float=0.02):
    self.cameras = cameras
    self.points = points
    self.scene = pyrender.Scene(bg_color=[1, 1, 1], ambient_light=[0.3, 0.3, 0.3, 1.0])

    # Calculate scene center and scale
    center = cameras.centers.cpu().numpy().mean(axis=0)

    self.marker_size = marker_size
    self.camera_markers = make_camera_markers(cameras, colors=(128, 128, 128), scale=self.marker_size)
    self.camera_node = self.scene.add(self.camera_markers)
    
    self.selected_markers = None

    self.point_mesh = pyrender.Mesh.from_points(points.points.cpu().numpy(), points.colors.cpu().numpy())
    self.point_node = self.scene.add(self.point_mesh)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    self.scene.add(light)

    # Find index of 'middle' camera (median position)
    median_idx = torch.median(cameras.centers, axis=0).indices[0].item()
    camera:camera_table.Camera = cameras[median_idx].item()

    viewport_size=(1920, 1080)

    self.scene.add_node(to_pyrender_camera(camera, viewport_size))
    self.space_pressed = False


    self.viewer = pyrender.Viewer(
        self.scene, 
        use_raymond_lighting=True, 
        view_center=center,  # Use calculated center
        viewport_size=viewport_size,
        point_size=4.0,
        run_in_thread=True,
        registered_keys={
            ' ': self._on_space,  # Space character as string
            'S': lambda _: None     # block default behavior
        }
    )

    self.trackball = self.viewer._trackball
    self.trackball._scale = self.camera_markers.scale

  def _on_space(self, viewer):
    self.space_pressed = True

  def wait_key(self):
    self.space_pressed = False
    while not self.space_pressed and self.viewer.is_active:
      time.sleep(0.01)

  def is_active(self):
    return self.viewer.is_active


  def replace_node(self, mesh:pyrender.Mesh, existing_node:Optional[pyrender.Node]=None):
    with self.viewer.render_lock:
        if existing_node is not None:
            self.scene.remove_node(existing_node)
        return self.scene.add(mesh)
    
    

  def colorize_cameras(self, colors:torch.Tensor):
    markers = make_camera_markers(self.cameras, colors=colors.cpu().numpy(), scale=self.marker_size)
    self.camera_node = self.replace_node(markers, self.camera_node)


  def show_selection(self, selected:Optional[torch.Tensor]=None):
    # show selected cameras as small sphere at camera centre
    if selected is not None:

      purple = torch.tensor([0.5, 0.0, 0.7], device=selected.device)
      red = torch.tensor([1.0, 0.0, 0.0], device=selected.device)

      color_map = torch.zeros(selected.shape[0], 3, device=selected.device)

      color_map[1:, :] = purple
      color_map[0, :] = red

      markers = make_spheres(radius=self.marker_size * 0.2, 
                             positions=self.cameras.centers[selected].cpu().numpy(), 
                             colors=color_map.cpu().numpy())
      
      self.selected_markers = self.replace_node(markers, self.selected_markers)

  def show_batch_selection(self, view_overlaps:torch.Tensor, selected:Optional[torch.Tensor]=None):

    t = view_overlaps - view_overlaps.min() 
    t = (t / t.max())

    green = torch.tensor([0.0, 1.0, 0.0], device=view_overlaps.device)
    color_map = green.unsqueeze(0) * t.unsqueeze(1)
        
    self.colorize_cameras(color_map)
    self.show_selection(selected)

