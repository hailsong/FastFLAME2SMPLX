from src.core import ftx
import cv2
import os

smplx_obj = "./data/smplx-addon.obj"
flame_obj = "./data/head_template.obj"
smplx_2_flame = "./data/SMPL-X__FLAME_vertex_ids.npy"
mean_texture = "./data/mean_texture.jpg"
face_vertex_ids = "./data/face_vertex_ids.npy"

flame_texture_path = "./input/song.png"
smplx_texture_path = "./input/smplx_sample_1.png"

output_dir = "./output/"

smplx_flame_uv_grid_path = f"./data/smplx_flame_uv_grid.npz"
flame_smplx_uv_grid_path = f"./data/flame_smplx_uv_grid.npz"

# FLAME to SMPLX conversion
texture_output = ftx.flame_smplx_texture_combine(
    flame_texture_path,
    smplx_texture_path,
    flame_obj,
    smplx_obj,
    smplx_2_flame,
    face_vertex_ids,
    uv_grid_load_path=None,
    uv_grid_save_path=flame_smplx_uv_grid_path
)
output_path = os.path.join(output_dir, "flame_to_smplx.png")
cv2.imwrite(output_path, texture_output)

# SMPLX to FLAME conversion
texture_output = ftx.smplx_flame_texture_combine(
    smplx_texture_path,
    mean_texture,
    smplx_obj,
    flame_obj,
    smplx_2_flame,
    face_vertex_ids,
    uv_grid_load_path=None,
    uv_grid_save_path=smplx_flame_uv_grid_path
)
output_path = os.path.join(output_dir, "smplx_to_flame.png")
cv2.imwrite(output_path, texture_output)
