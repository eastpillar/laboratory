import os
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from torchvision import transforms
import re
from network2 import TokenHPE
import utils

# ---------- 설정 ----------
LIST_PATH  = "/Desktop/hdd/adj_test/dataset/front_view_path.txt"
FACE_RESIZE = (224, 224)
MIN_CONF    = 0.7
AD          = 0.7        # bbox padding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
print('current cuda device:', torch.cuda.current_device())
THRESH_DIFF = 18.0
MAX_MOVE_FRAC = 0.01
DETECT_WIN_W = 1300
DETECT_WIN_H = 800

def _frame_index_from_path(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r'(\d+)$', stem)
    return int(m.group(1)) if m else 0

def _split_scenario_parts_from_image(img_path: str):
    d0 = os.path.dirname(img_path)
    cap_dir = os.path.dirname(d0)
    scenario_root = os.path.dirname(cap_dir)
    subdir = os.path.basename(d0)
    filename = os.path.basename(img_path)
    return scenario_root, subdir, filename

def _split_scenario_parts_from_folder(folder_path: str):
    cap_dir = os.path.dirname(folder_path)
    scenario_root = os.path.dirname(cap_dir)
    subdir = os.path.basename(folder_path)
    return scenario_root, subdir

def _map_tiff_output_path_from_folder(folder_path: str) -> str:
    scenario_root, subdir = _split_scenario_parts_from_folder(folder_path)
    out_dir = os.path.join(scenario_root, "yrp-images")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{subdir}.tiff")

def _map_vis_output_path(src_img_path: str) -> str:
    scenario_root, subdir, filename = _split_scenario_parts_from_image(src_img_path)
    out_dir = os.path.join(scenario_root, "visImage", subdir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


TokenHPE_path = './TokenHPEv1-ViTB-224_224-lyr3.tar'
HPE_model_ckpt = torch.load(TokenHPE_path, map_location='cpu')

model = TokenHPE(num_ori_tokens=9, depth=3, heads=8, embedding='sine', dim=128, inference_view=False).to(DEVICE).eval()
model.load_state_dict(HPE_model_ckpt['model_state_dict'])

detector = MTCNN(keep_all=True, device=DEVICE)

face_tf = transforms.Compose([
    transforms.Resize(FACE_RESIZE[0]),
    transforms.CenterCrop(FACE_RESIZE[0]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def read_list(list_path):
    paths = []
    with open(list_path, 'r') as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    return paths

def group_by_cap_subdir(paths):
    buckets = {}
    for p in paths:
        parent = os.path.dirname(p)
        buckets.setdefault(parent, []).append(p)

    for k in buckets:
        buckets[k].sort(key=_frame_index_from_path)

    return buckets

def run_tokenhpe_on_face(img_bgr, box):
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2-x1, y2-y1
    # padding crop
    xw1 = max(int(x1 - AD * bw), 0)
    yw1 = max(int(y1 - AD * bh), 0)
    xw2 = min(int(x2 + AD * bw), W-1)
    yw2 = min(int(y2 + AD * bh), H-1)
    if xw2 <= xw1 or yw2 <= yw1:
        return None

    crop = img_bgr[yw1:yw2+1, xw1:xw2+1, :]
    face_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    save_dir = '../test_images'
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(run_tokenhpe_on_face, "q"):
        run_tokenhpe_on_face.q = 0
    save_path = os.path.join(save_dir, f"{run_tokenhpe_on_face.q:06d}.jpg")
    cv2.imwrite(save_path, crop)
    run_tokenhpe_on_face.q += 1

    face_tensor = face_tf(face_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_R, _ = model(face_tensor)
        euler_rad = utils.compute_euler_angles_from_rotation_matrices(pred_R, use_gpu=(DEVICE.type=='cuda'))
        euler_deg = euler_rad * 180.0 / np.pi
        pitch_deg = float(euler_deg[0,0].item())
        yaw_deg   = float(euler_deg[0,1].item())
        roll_deg  = float(euler_deg[0,2].item())

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    axis_len = max(40, int(0.6 * max(bw, bh)))

    return (yaw_deg, roll_deg, pitch_deg, cx, cy, axis_len)

def pick_best_face(img_bgr, boxes, probs, prev_box=None):
    if boxes is None or len(boxes) == 0:
        return None, None

    H, W = img_bgr.shape[:2]

    best_idx, best_y2 = 0, -1e9
    for i, b in enumerate(boxes):
        y2 = b[3]
        if y2 > best_y2:
            best_y2 = y2
            best_idx = i

    sel_box = boxes[best_idx]
    sel_prob = float(probs[best_idx]) if probs is not None else None

    max_move_px = MAX_MOVE_FRAC * min(H, W)
    sel_box = _clamp_box_movement(sel_box, prev_box, max_move_px, H, W)

    return sel_box, sel_prob

def fill_ypr_for_folder(img_paths):
    N = len(img_paths)
    ypr = np.zeros((3, N), dtype=np.float32)
    centers = [None] * N
    sizes = [None] * N

    prev_box = None
    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]

        win_w = min(DETECT_WIN_W, W)
        win_h = min(DETECT_WIN_H, H)
        x0 = (W - win_w) // 2
        y0 = (H - win_h) // 2
        x1 = x0 + win_w
        y1 = y0 + win_h

        crop_bgr = img[y0:y1, x0:x1]
        pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        boxes, probs, _ = detector.detect(pil_crop, landmarks=True)

        if boxes is not None and len(boxes) > 0:
            boxes_full = []
            probs_full = []
            for b, pr in zip(boxes, (probs if probs is not None else [None]*len(boxes))):
                bx1, by1, bx2, by2 = b
                fb = [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]
                if fb[0] < x0 or fb[1] < y0 or fb[2] > x1 or fb[3] > y1:
                    continue
                boxes_full.append(fb)
                probs_full.append(float(pr) if pr is not None else None)
            if len(boxes_full) == 0:
                boxes_full = None
                probs_full = None
        else:
            boxes_full = None
            probs_full = None

        box, conf = pick_best_face(img, boxes_full, probs_full, prev_box=prev_box)

        if (box is None) or (conf is not None and conf < MIN_CONF):
            if prev_box is None:
                continue
            else:
                box = prev_box


        out = run_tokenhpe_on_face(img, box)
        if out is None:
            continue
        yaw_deg, roll_deg, pitch_deg, cx, cy, axis_len = out
        ypr[:, i] = np.array([yaw_deg, roll_deg, pitch_deg], dtype=np.float32)
        centers[i] = (cx, cy)
        sizes[i] = axis_len
        prev_box = box

    return ypr, centers, sizes



def postprocess_ypr(ypr, centers):
    ypr_proc = ypr.copy()
    C, N = ypr_proc.shape
    orig = ypr.copy()

    for i in range(N - 1):
        diff = np.abs(orig[:, i] - orig[:, i + 1]).sum(0)
        if diff > THRESH_DIFF:
            ypr_proc[:, i] = 0.0
            ypr_proc[:, i + 1] = 0.0

    for i in range(N):
        p0 = max(0, i - 5)
        past_any_zero = False
        for j in range(p0, i):
            if np.allclose(ypr_proc[:, j], 0.0):
                past_any_zero = True
                break
        if not past_any_zero:
            continue

        f1 = min(N - 1, i + 10)
        future_any_zero = False
        for j in range(i, f1 + 1):
            if np.allclose(ypr_proc[:, j], 0.0):
                future_any_zero = True
                break
        if not future_any_zero:
            continue

        ypr_proc[:, i] = 0.0

    PIX_THR = 20
    anchor = None

    i = 0
    while i < N:
        c = centers[i]
        if c is not None:
            if anchor is None:
                anchor = c
            dx = abs(c[0] - anchor[0])
            dy = abs(c[1] - anchor[1])
            if max(dx, dy) > PIX_THR:
                j = i + 1
                while j < N:
                    cj = centers[j]
                    if cj is None:
                        ypr_proc[:, j] = 0.0
                        j += 1
                        continue
                    dxj = abs(cj[0] - anchor[0])
                    dyj = abs(cj[1] - anchor[1])
                    if max(dxj, dyj) > PIX_THR:
                        ypr_proc[:, j] = 0.0
                        j += 1
                        continue
                    anchor = cj
                    break
                i = j
                continue
            else:
                anchor = c
        i += 1

    return ypr_proc

def _clamp_box_movement(box, prev_box, max_move_px, H, W):
    if prev_box is None or box is None:
        return box

    x1, y1, x2, y2 = [float(v) for v in box]
    px1, py1, px2, py2 = [float(v) for v in prev_box]

    cx  = 0.5 * (x1 + x2);  cy  = 0.5 * (y1 + y2)
    pcx = 0.5 * (px1 + px2); pcy = 0.5 * (py1 + py2)

    dx = cx - pcx
    dy = cy - pcy
    dist = (dx*dx + dy*dy) ** 0.5
    if dist <= max_move_px or dist == 0:
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    scale = max_move_px / dist
    cx_clamp = pcx + dx * scale
    cy_clamp = pcy + dy * scale

    w = (x2 - x1)
    h = (y2 - y1)
    nx1 = int(round(cx_clamp - 0.5 * w))
    ny1 = int(round(cy_clamp - 0.5 * h))
    nx2 = int(round(cx_clamp + 0.5 * w))
    ny2 = int(round(cy_clamp + 0.5 * h))

    nx1 = max(0, min(nx1, W - 1))
    ny1 = max(0, min(ny1, H - 1))
    nx2 = max(0, min(nx2, W - 1))
    ny2 = max(0, min(ny2, H - 1))
    if nx2 <= nx1 or ny2 <= ny1:
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
    return [nx1, ny1, nx2, ny2]

def save_tiff_for_folder(folder_path, ypr_final):
    tiff_path = _map_tiff_output_path_from_folder(folder_path)

    try:
        import imageio.v2 as imageio
        imageio.imwrite(tiff_path, ypr_final.astype(np.float32), format='TIFF')
    except Exception:
        try:
            from tifffile import imwrite
            imwrite(tiff_path, ypr_final.astype(np.float32))
        except Exception:
            np.save(tiff_path + ".npy", ypr_final.astype(np.float32))
            print(f"[경고] TIFF 저장 실패 → NPY 저장: {tiff_path+'.npy'}")
            return
    print(f"[저장] {tiff_path} (shape={ypr_final.shape})")


def draw_images_for_folder(folder_path, img_paths, ypr_final, centers, sizes):
    for i, p in enumerate(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue

        yaw, roll, pitch = ypr_final[:, i].tolist()

        out_path = _map_vis_output_path(p)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if centers[i] is not None and sizes[i] is not None and (yaw != 0 or roll != 0 or pitch != 0):
            cx, cy = centers[i]
            axis_len = int(sizes[i])
            utils.draw_axis(img, yaw=yaw, pitch=pitch, roll=roll, tdx=cx, tdy=cy, size=axis_len)

        cv2.imwrite(out_path, img)


def main():
    paths = read_list(LIST_PATH)
    groups = group_by_cap_subdir(paths)

    print(f"총 {len(groups)}개 폴더 그룹 처리")
    for folder, img_paths in tqdm(groups.items()):
        ypr_raw, centers, sizes = fill_ypr_for_folder(img_paths)

        ypr_final = postprocess_ypr(ypr_raw, centers)
        save_tiff_for_folder(folder, ypr_final)

        draw_images_for_folder(folder, img_paths, ypr_final, centers, sizes)

if __name__ == "__main__":
    main()

