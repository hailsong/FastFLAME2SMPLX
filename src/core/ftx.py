import os
from typing import Optional, Tuple, Literal, Dict

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import tqdm

# project utils
from .file_op import (
    read_vertex_faces_id_from_obj,
    read_uv_faces_id_from_obj,
    read_uv_coordinates_from_obj,
    read_vertex_from_obj,
)


# ---------------------- Rasterization & UV Grid ----------------------

def _rasterize_barycentric(uv, faces, H, W, device):
    """
    uv: [V,2] in [0,1]
    faces: [F,3] int
    returns:
      pix_to_face: [H,W] int64 (-1 for background)
      bary_coords: [H,W,3] float32
    """
    ys = (torch.arange(H, device=device) + 0.5) / H
    xs = (torch.arange(W, device=device) + 0.5) / W
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
    pix = torch.stack([grid_x, grid_y], dim=-1)             # [H,W,2]

    pix_to_face = torch.full((H, W), -1, dtype=torch.int64, device=device)
    bary_coords = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    for fi, tri in enumerate(faces):
        tri_uv = uv[tri].to(device)
        min_uv, _ = tri_uv.min(dim=0)
        max_uv, _ = tri_uv.max(dim=0)
        y0 = int((min_uv[1] * H).clamp(0, H-1).item())
        y1 = int((max_uv[1] * H).clamp(0, H-1).item()) + 1
        x0 = int((min_uv[0] * W).clamp(0, W-1).item())
        x1 = int((max_uv[0] * W).clamp(0, W-1).item()) + 1

        subpix = pix[y0:y1, x0:x1].reshape(-1, 2)
        v0 = tri_uv[1] - tri_uv[0]
        v1 = tri_uv[2] - tri_uv[0]
        v2 = subpix - tri_uv[0]
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0).sum(-1)
        d21 = (v2 * v1).sum(-1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        bc = torch.stack([u, v, w], dim=-1)

        mask = (bc >= 0).all(dim=-1) & (bc <= 1).all(dim=-1)
        if not mask.any():
            continue

        coords = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        ys_idx = coords // (x1 - x0)
        xs_idx = coords % (x1 - x0)
        global_ys = ys_idx + y0
        global_xs = xs_idx + x0
        pix_to_face[global_ys, global_xs] = fi
        bary_coords[global_ys, global_xs] = bc[mask]

    return pix_to_face, bary_coords


def calculate_uv_grid(
    mapping: Dict[int, int],
    src_uv,        # numpy [V_s,2] in [0,1]
    src_faces,     # numpy [F_s,3] int
    tgt_uv,        # numpy [V_t,2]
    tgt_faces,     # numpy [F_t,3]
    H: int,
    W: int,
    device: torch.device,
):
    """
    mapping: dict[source_face_id -> target_face_id] (or inverse, as long as consistent).
    Uses src_ids = list(mapping.keys()), tgt_ids = [mapping[s] for s in src_ids].
    """
    src_ids = list(mapping.keys())
    tgt_ids = [mapping[s] for s in src_ids]

    uv_s = torch.from_numpy(src_uv).float().to(device)
    uv_t = torch.from_numpy(tgt_uv).float().to(device)
    faces_s = torch.from_numpy(src_faces).long().to(device)
    faces_t = torch.from_numpy(tgt_faces).long().to(device)

    pix_to_face, bary = _rasterize_barycentric(uv_t, faces_t[tgt_ids], H, W, device)
    uv_grid = torch.zeros((1, H, W, 2), device=device)

    for i, sid in enumerate(tqdm.tqdm(src_ids, desc="Blending triangles")):
        mask = pix_to_face == i
        if not mask.any():
            continue
        bc = bary[mask]
        tri_uv = uv_s[faces_s[sid]]
        pix_uv = (bc.unsqueeze(-1) * tri_uv).sum(dim=1)    # [N,2] in [0,1]
        pix_uv = pix_uv * 2.0 - 1.0                        # [-1,1]
        uv_grid[0, mask, :] = pix_uv

    return uv_grid, pix_to_face


def save_uv_grid(filepath: str, uv_grid: torch.Tensor, pix_to_face: torch.Tensor):
    np_uv   = uv_grid.detach().cpu().numpy().astype(np.float32)
    np_pix_to_face = pix_to_face.detach().cpu().numpy().astype(np.int64)
    np.savez(filepath, uv_grid=np_uv, pix_to_face=np_pix_to_face)


def load_uv_grid(filepath: str, device: torch.device):
    data = np.load(filepath)
    uv_grid = torch.from_numpy(data["uv_grid"]).to(device)         # [1,H,W,2]
    pix_to_face = torch.from_numpy(data["pix_to_face"]).to(device) # [H,W]
    pix_to_face = pix_to_face.float()
    return uv_grid, pix_to_face


# ---------------------- Cross-correspondence (SMPL-X ↔ FLAME) ----------------------

def get_smplx_flame_crossrespondence_face_ids(
    smplx_template_obj,
    flame_template_obj,
    smplx_flame_vertex_ids,
    smplx_face_indexes=None
):
    s_f_ids = read_vertex_faces_id_from_obj(smplx_template_obj)
    s_f_uvs = read_uv_faces_id_from_obj(smplx_template_obj)
    s_uv = read_uv_coordinates_from_obj(smplx_template_obj)
    s_uv[:, 1] = 1 - s_uv[:, 1]

    f_verts = read_vertex_from_obj(flame_template_obj)
    f_f_ids = read_vertex_faces_id_from_obj(flame_template_obj)
    f_f_uvs = read_uv_faces_id_from_obj(flame_template_obj)
    f_uv = read_uv_coordinates_from_obj(flame_template_obj)
    f_uv[:, 1] = 1 - f_uv[:, 1]

    sf_ids = np.load(smplx_flame_vertex_ids)  # len == len(FLAME verts), values = SMPL-X vert id (or -1)
    if smplx_face_indexes is not None:
        face_ids = np.load(smplx_face_indexes)
        for j in range(len(sf_ids)):
            if sf_ids[j] not in face_ids:
                sf_ids[j] = -1

    f2s = {i: sf_ids[i] for i in range(len(f_verts))}
    smplx_faces = {f"{v0}_{v1}_{v2}": idx for idx, (v0, v1, v2) in enumerate(s_f_ids)}

    mapping = {}
    for fid, (v0, v1, v2) in enumerate(f_f_ids):
        key = f"{f2s[v0]}_{f2s[v1]}_{f2s[v2]}"
        if key in smplx_faces:
            mapping[fid] = smplx_faces[key]

    return mapping, s_f_uvs, s_uv, f_f_uvs, f_uv


# ---------------------- Main transfer (lazy mapping inside) ----------------------

def _barycentric_texture_transfer(
    src_texture_path: str,
    tgt_texture_path: str,
    src_size: tuple,
    tgt_size: tuple,
    mapping: Optional[Dict[int, int]] = None,
    src_uv: Optional[np.ndarray] = None,
    src_faces: Optional[np.ndarray] = None,
    tgt_uv: Optional[np.ndarray] = None,
    tgt_faces: Optional[np.ndarray] = None,
    uv_grid_load_path: Optional[str] = None,
    uv_grid_save_path: Optional[str] = None,
    *,
    # used only when mapping=None
    direction: Optional[Literal["flame->smplx", "smplx->flame"]] = None,
    smplx_template_obj: Optional[str] = None,
    flame_template_obj: Optional[str] = None,
    smplx_flame_vertex_ids: Optional[str] = None,
    smplx_face_indexes: Optional[str] = None,
) -> np.ndarray:
    """
    If uv_grid exists: load and sample.
    Else:
      - if mapping provided: build grid directly
      - if mapping is None: compute correspondence here, then build grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_img = cv2.imread(src_texture_path)
    if (src_img.shape[1], src_img.shape[0]) != src_size:
        src_img = cv2.resize(src_img, src_size)

    tgt_img = cv2.imread(tgt_texture_path)
    if (tgt_img.shape[1], tgt_img.shape[0]) != tgt_size:
        tgt_img = cv2.resize(tgt_img, tgt_size)

    Ht, Wt = tgt_img.shape[:2]

    src_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_t = torch.from_numpy(src_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    if uv_grid_load_path and os.path.exists(uv_grid_load_path):
        uv_grid, pix_to_face = load_uv_grid(uv_grid_load_path, device)
        print(f"[TextureTransfer] Loaded UV grid from {uv_grid_load_path}")
    else:
        if mapping is None:
            if direction is None:
                raise ValueError("direction must be provided when mapping=None.")
            if not (smplx_template_obj and flame_template_obj and smplx_flame_vertex_ids):
                raise ValueError("Provide smplx_template_obj, flame_template_obj, smplx_flame_vertex_ids.")

            map_ff2ss, s_uvf, s_uv_all, f_uvf, f_uv_all = get_smplx_flame_crossrespondence_face_ids(
                smplx_template_obj, flame_template_obj, smplx_flame_vertex_ids, smplx_face_indexes
            )

            if direction == "flame->smplx":
                mapping = map_ff2ss
                src_uv = f_uv_all
                src_faces = f_uvf
                tgt_uv = s_uv_all
                tgt_faces = s_uvf
            elif direction == "smplx->flame":
                mapping = {v: k for k, v in map_ff2ss.items()}
                src_uv = s_uv_all
                src_faces = s_uvf
                tgt_uv = f_uv_all
                tgt_faces = f_uvf
            else:
                raise ValueError(f"Unknown direction: {direction}")

        if any(x is None for x in [src_uv, src_faces, tgt_uv, tgt_faces]):
            raise ValueError("src_uv/src_faces/tgt_uv/tgt_faces must be provided or computable when uv_grid is not loaded.")

        uv_grid, pix_to_face = calculate_uv_grid(mapping, src_uv, src_faces, tgt_uv, tgt_faces, Ht, Wt, device)

        if uv_grid_save_path:
            save_uv_grid(uv_grid_save_path, uv_grid, pix_to_face)
            print(f"[TextureTransfer] Saved UV grid to {uv_grid_save_path}")

    sampled = F.grid_sample(src_t, uv_grid, mode='bilinear', align_corners=True)

    tgt_rgb = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
    tgt_t = torch.from_numpy(tgt_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    mask = (pix_to_face >= 0).float().unsqueeze(0).unsqueeze(0)
    out = sampled * mask + tgt_t * (1 - mask)

    out_np = (out[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr


# ---------------------- Convenience wrappers ----------------------

def flame_smplx_texture_combine(
    flame_texture_path: str,
    smplx_texture_path: str,
    flame_obj: str = None,
    smplx_obj: str = None,
    smplx_flame_vertex_ids: str = None,
    smplx_face_indexes: Optional[str] = None,
    uv_grid_load_path: Optional[str] = None,
    uv_grid_save_path: Optional[str] = None,
    src_size: Tuple[int, int] = (2048, 2048),
    tgt_size: Tuple[int, int] = (4096, 4096),
) -> np.ndarray:
    return _barycentric_texture_transfer(
        src_texture_path=flame_texture_path,
        tgt_texture_path=smplx_texture_path,
        src_size=src_size,
        tgt_size=tgt_size,
        mapping=None,
        uv_grid_load_path=uv_grid_load_path,
        uv_grid_save_path=uv_grid_save_path,
        direction="flame->smplx",
        smplx_template_obj=smplx_obj,
        flame_template_obj=flame_obj,
        smplx_flame_vertex_ids=smplx_flame_vertex_ids,
        smplx_face_indexes=smplx_face_indexes,
    )


def smplx_flame_texture_combine(
    smplx_texture_path: str,
    flame_texture_path: str,
    smplx_obj: str = None,
    flame_obj: str = None,
    smplx_flame_vertex_ids: str = None,
    smplx_face_indexes: Optional[str] = None,
    uv_grid_load_path: Optional[str] = None,
    uv_grid_save_path: Optional[str] = None,
    src_size: Tuple[int, int] = (4096, 4096),
    tgt_size: Tuple[int, int] = (2048, 2048),
) -> np.ndarray:
    return _barycentric_texture_transfer(
        src_texture_path=smplx_texture_path,
        tgt_texture_path=flame_texture_path,
        src_size=src_size,
        tgt_size=tgt_size,
        mapping=None,
        uv_grid_load_path=uv_grid_load_path,
        uv_grid_save_path=uv_grid_save_path,
        direction="smplx->flame",
        smplx_template_obj=smplx_obj,
        flame_template_obj=flame_obj,
        smplx_flame_vertex_ids=smplx_flame_vertex_ids,
        smplx_face_indexes=smplx_face_indexes,
    )
