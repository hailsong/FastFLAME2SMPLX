import cv2
import numpy as np


def load_obj_uv_faces(obj_path):
    uv_coords = []
    faces = []

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('vt '):  # texture coords
                parts = line.strip().split()
                uv_coords.append([float(parts[1]), float(parts[2])])  # vt u v
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                face = []
                for p in parts:
                    if '/' in p:
                        tokens = p.split('/')
                        if len(tokens) >= 2:
                            vt_idx = int(tokens[1]) - 1  # 0-based indexing
                            face.append(vt_idx)
                if len(face) == 3:
                    faces.append(face)

    return np.array(uv_coords), np.array(faces)


def draw_uv_wireframe(uv_coords, faces, H=4096, W=4096, color=(255, 255, 255)):
    uv_img = np.zeros((H, W, 3), dtype=np.uint8)

    uv_coords = uv_coords.copy()
    uv_coords[:, 0] *= (W - 1)
    uv_coords[:, 1] *= (H - 1)
    uv_coords[:, 1] = H - 1 - uv_coords[:, 1]  # Flip Y-axis

    for face in faces:
        pts = uv_coords[face].astype(np.int32)
        cv2.polylines(uv_img, [pts], isClosed=True, color=color, thickness=1)

    return uv_img


if __name__ == "__main__":
    # Edit these paths
    flame_obj_path = "../data/head_template.obj"
    smplx_obj_path = "../data/smplx-addon.obj"

    flame_uv, flame_faces = load_obj_uv_faces(flame_obj_path)
    smplx_uv, smplx_faces = load_obj_uv_faces(smplx_obj_path)

    flame_wireframe = draw_uv_wireframe(flame_uv, flame_faces)
    smplx_wireframe = draw_uv_wireframe(smplx_uv, smplx_faces)
    flame_wireframe = cv2.resize(flame_wireframe, (2048, 2048), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("../data/flame_uv_wireframe.png", flame_wireframe)
    cv2.imwrite("../data/smplx_uv_wireframe.png", smplx_wireframe)
