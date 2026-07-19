"""
PositionalRegionSwap — позиционный свап лиц ПО РЕГИОНАМ (порт эталона test_pose_matrix_v4._swap_all).

Алгоритм (валидирован матрицей поз, 94 позы, 15-17.07.2026):
  детект insightface buffalo_l -> адаптивный передетект det_thresh 0.30 (если лиц < персонажей)
  -> партиция лиц по полу -> X-сорт внутри пола -> назначение рефов В ПОРЯДКЕ РЕГИОНОВ
  -> inswapper_128 -> BJ mouth-preserve (эллипс по kps) -> кросс-гендер fallback для осиротевших
  -> codeformer.onnx restore (feather-вклейка, blend 80).

insightface ВЕНДОРЕН в vendor/ (pip-пакет тянет CPU-onnxruntime и ломает GPU у FaceFusion — НЕ ставить pip'ом).
Модели: models/buffalo_l/{det_10g,genderage,w600k_r50}.onnx + models/codeformer.onnx (манифест),
inswapper_128.onnx берётся из Facefusion_comfyui/models (уже в контейнере) или models/.

Вход plan (JSON, порядок = порядок регионов):
  [{"sex": "M"|"F", "is_bj": false, "ref": 1|2|3|null, "skip": false}, ...]
"ref" — номер входа ref_N с рефом этого слота. "skip": true (или ref=null) — персонаж БЕЗ рефа:
он УЧАСТВУЕТ в позиционной раскладке (его лицо занимает свою позицию и защищается от fallback),
но не свапается — иначе частичный свап сажал бы реф на чужое лицо.
"""
import os
import sys
import json

import numpy as np
import cv2
# torch импортируется лениво в конвертерах IMAGE (в контейнере есть; локальный smoke без torch)

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(NODE_DIR, "vendor")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

MODELS_DIR = os.path.join(NODE_DIR, "models")
CODEFORMER_PATH = os.path.join(MODELS_DIR, "codeformer.onnx")
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CODEFORMER_FIDELITY_WEIGHT = 0.5

_app = None
_swapper = None
_restorer = None


def _get_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        print("[PositionalSwap] init FaceAnalysis(buffalo_l) ...")
        _app = FaceAnalysis(name="buffalo_l", root=NODE_DIR,
                            allowed_modules=["detection", "recognition", "genderage"],
                            providers=PROVIDERS)
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def _find_inswapper():
    cands = [
        os.path.join(MODELS_DIR, "inswapper_128.onnx"),
        os.path.join(os.path.dirname(NODE_DIR), "Facefusion_comfyui", "models", "inswapper_128.onnx"),
    ]
    for c in cands:
        if os.path.exists(c) and os.path.getsize(c) > 100_000_000:
            return c
    raise RuntimeError(f"[PositionalSwap] inswapper_128.onnx not found in {cands}")


def _get_swapper():
    global _swapper
    if _swapper is None:
        import insightface
        path = _find_inswapper()
        print(f"[PositionalSwap] load inswapper: {path}")
        _swapper = insightface.model_zoo.get_model(path, providers=PROVIDERS)
    return _swapper


def _get_restorer():
    global _restorer
    if _restorer is None:
        import onnxruntime as ort
        if not os.path.exists(CODEFORMER_PATH):
            print(f"[PositionalSwap] WARN: codeformer.onnx not found ({CODEFORMER_PATH}) — restore skipped")
            _restorer = False
        else:
            print(f"[PositionalSwap] load codeformer: {CODEFORMER_PATH}")
            _restorer = ort.InferenceSession(CODEFORMER_PATH, providers=PROVIDERS)
    return _restorer


# ── restore (порт test_local_insightface_swap) ───────────────────────────────
_FEATHER_MASK_512 = None


def _get_feather_mask(size=512, border_frac=0.08, blur_frac=0.06):
    global _FEATHER_MASK_512
    if _FEATHER_MASK_512 is not None:
        return _FEATHER_MASK_512
    border = int(size * border_frac)
    m = np.zeros((size, size), dtype=np.float32)
    m[border:size - border, border:size - border] = 1.0
    k = int(size * blur_frac)
    if k % 2 == 0:
        k += 1
    _FEATHER_MASK_512 = cv2.GaussianBlur(m, (k, k), 0)
    return _FEATHER_MASK_512


def _run_restorer(session, aligned_bgr):
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    inp = rgb.transpose(2, 0, 1)[None].astype(np.float32)
    input_names = {i.name for i in session.get_inputs()}
    feeds = {"input": inp}
    if "weight" in input_names:
        feeds["weight"] = np.array(CODEFORMER_FIDELITY_WEIGHT, dtype=np.float64)
    out = session.run(["output"], feeds)[0]
    out = out[0].transpose(1, 2, 0)
    out = ((out * 0.5 + 0.5).clip(0, 1) * 255.0).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _restore_face(canvas_bgr, face, session, blend_pct):
    from insightface.utils import face_align
    M = face_align.estimate_norm(face.kps, image_size=512, mode="arcface")
    aligned = cv2.warpAffine(canvas_bgr, M, (512, 512), borderValue=0.0)
    restored_aligned = _run_restorer(session, aligned)
    h, w = canvas_bgr.shape[:2]
    Minv = cv2.invertAffineTransform(M)
    warped_restored = cv2.warpAffine(restored_aligned, Minv, (w, h), borderMode=cv2.BORDER_REPLICATE)
    warped_mask = cv2.warpAffine(_get_feather_mask(), Minv, (w, h))
    alpha = (warped_mask * (blend_pct / 100.0))[:, :, None]
    out = canvas_bgr.astype(np.float32) * (1 - alpha) + warped_restored.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def _apply_restore(swapped_bgr, app, session, blend_pct):
    faces = app.get(swapped_bgr)
    print(f"[PositionalSwap] restore: faces={len(faces)}, blend={blend_pct}")
    out = swapped_bgr
    for f in faces:
        out = _restore_face(out, f, session, blend_pct)
    return out


# ── BJ mouth-preserve (порт test_sc01_imaginator_v2._preserve_mouth) ─────────
def _preserve_mouth(orig_bgr, swapped_bgr, face):
    try:
        kps = face.kps
        nose = np.asarray(kps[2], np.float32)
        lm, rm = np.asarray(kps[3], np.float32), np.asarray(kps[4], np.float32)
        cx, cy = (lm[0] + rm[0]) / 2, (lm[1] + rm[1]) / 2
        mouth_w = float(np.linalg.norm(rm - lm))
        ax = max(10, int(mouth_w * 1.0))
        ay = max(10, int(np.linalg.norm(np.array([cx, cy]) - nose) * 1.4))
        cyc = int(cy + ay * 0.3)
        center = (int(cx), cyc)
    except Exception:
        x0, y0, x1, y1 = [int(v) for v in face.bbox]
        center = ((x0 + x1) // 2, int(y0 + (y1 - y0) * 0.75))
        ax, ay = (x1 - x0) // 3, (y1 - y0) // 5
    mask = np.zeros(orig_bgr.shape[:2], np.float32)
    cv2.ellipse(mask, center, (ax, ay), 0, 0, 360, 1.0, -1)
    k = max(3, (ax // 2) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
    out = swapped_bgr.astype(np.float32) * (1 - mask) + orig_bgr.astype(np.float32) * mask
    return out.clip(0, 255).astype(np.uint8)


# ── ядро: порт _swap_all (test_pose_matrix_v4) ───────────────────────────────
def _prepare_source_face(app, ref_bgr, idx):
    faces = app.get(ref_bgr)
    if not faces:
        raise RuntimeError(f"[PositionalSwap] no face in ref_{idx + 1}")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _swap_all(scene_bgr, plan, refs_bgr_by_slot, restore_blend):
    """plan = [{"sex","is_bj","skip"}...] в порядке регионов; refs_bgr_by_slot[i] — реф слота i (или None)."""
    app, swapper = _get_app(), _get_swapper()

    faces = app.get(scene_bgr)
    if len(faces) < len(plan):                      # адаптивный передетект трудных лиц
        _th = app.det_model.det_thresh
        try:
            app.det_model.det_thresh = 0.30
            f2 = app.get(scene_bgr)
            if len(f2) > len(faces):
                print(f"[PositionalSwap] redetect(0.30): {len(faces)} -> {len(f2)}")
                faces = f2
        finally:
            app.det_model.det_thresh = _th
    print(f"[PositionalSwap] faces={len(faces)}, plan={len(plan)}")

    faces_by_sex = {"M": [], "F": []}
    for f in faces:
        sx = getattr(f, "sex", None) or ("M" if getattr(f, "gender", 0) == 1 else "F")
        faces_by_sex[sx].append(f)
    for sx in faces_by_sex:
        faces_by_sex[sx].sort(key=lambda f: f.bbox[0])

    chars_by_sex = {"M": [], "F": []}
    for i, slot in enumerate(plan):
        chars_by_sex[slot["sex"]].append((i, slot))

    sources = {}
    result = scene_bgr.copy()
    n = 0
    used = set()

    def do(char_idx, slot, tgt, label):
        nonlocal result, n
        if slot.get("skip") or refs_bgr_by_slot[char_idx] is None:
            # без рефа: позицию занимает, лицо защищено от fallback, но не свапаем
            print(f"[PositionalSwap] {label} x0={tgt.bbox[0]:.0f} slot{char_idx + 1} SKIP (no ref)")
            used.add(id(tgt))
            return
        if char_idx not in sources:
            sources[char_idx] = _prepare_source_face(app, refs_bgr_by_slot[char_idx], char_idx)
        src = sources[char_idx]
        print(f"[PositionalSwap] {label} x0={tgt.bbox[0]:.0f} <- slot{char_idx + 1}"
              f"{' +BJ' if slot.get('is_bj') else ''}")
        sw = swapper.get(result, tgt, src, paste_back=True)
        if slot.get("is_bj"):
            sw = _preserve_mouth(result, sw, tgt)
        result = sw
        n += 1
        used.add(id(tgt))

    leftover = []
    for sx in ("M", "F"):
        chs, fcs = chars_by_sex[sx], faces_by_sex.get(sx, [])
        for i in range(len(chs)):
            if i < len(fcs):
                do(chs[i][0], chs[i][1], fcs[i], f"swap {sx}[{i}]")
            else:
                leftover.append(chs[i])
    free = [f for f in sorted(faces, key=lambda f: f.bbox[0]) if id(f) not in used]
    if leftover and free:
        print(f"[PositionalSwap] fallback: {len(leftover)} chars <- {len(free)} faces (ignore sex)")
        for (ci, slot), tgt in zip(leftover, free):
            do(ci, slot, tgt, "swap fallback")

    if n and restore_blend > 0:
        session = _get_restorer()
        if session:
            result = _apply_restore(result, app, session, restore_blend)
    elif not n:
        print("[PositionalSwap] nothing swapped")
    return result


# ── ComfyUI обвязка ──────────────────────────────────────────────────────────
def _img_to_bgr(image_tensor):
    """Comfy IMAGE (B,H,W,C float 0..1 RGB) -> BGR uint8 (первый кадр батча)."""
    arr = (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_img(bgr):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb)[None]


class PositionalRegionSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "plan": ("STRING", {"default": '[{"sex":"F","is_bj":false}]', "multiline": True}),
                "restore_blend": ("INT", {"default": 80, "min": 0, "max": 100}),
            },
            "optional": {
                "ref_1": ("IMAGE",),
                "ref_2": ("IMAGE",),
                "ref_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("swapped_image",)
    FUNCTION = "execute"
    CATEGORY = "PositionalSwap"

    def execute(self, image, plan, restore_blend, ref_1=None, ref_2=None, ref_3=None):
        slots = json.loads(plan)
        ref_inputs = {1: ref_1, 2: ref_2, 3: ref_3}
        refs_bgr_by_slot = []
        for i, slot in enumerate(slots):
            if slot.get("skip"):
                refs_bgr_by_slot.append(None)
                continue
            rn = slot.get("ref", i + 1)     # дефолт: слот i ↔ ref_{i+1}
            img_ref = ref_inputs.get(rn)
            if img_ref is None:
                raise ValueError(f"[PositionalSwap] slot {i + 1} expects ref_{rn} but it is not connected")
            refs_bgr_by_slot.append(_img_to_bgr(img_ref))
        scene_bgr = _img_to_bgr(image)
        out = _swap_all(scene_bgr, slots, refs_bgr_by_slot, restore_blend)
        return (_bgr_to_img(out),)


NODE_CLASS_MAPPINGS = {"PositionalRegionSwap": PositionalRegionSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"PositionalRegionSwap": "Positional Region Swap (matrix)"}
