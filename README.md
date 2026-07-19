# ComfyUI-PositionalSwap

Positional face swap by REGIONS (pose-matrix driven). Port of the validated local swap
(`test_pose_matrix_v4._swap_all`): buffalo_l detect -> adaptive redetect 0.30 -> sex partition ->
X-sort -> region-order assignment -> inswapper_128 -> BJ mouth-preserve -> cross-gender fallback ->
codeformer.onnx restore.

insightface is VENDORED in `vendor/` (do NOT pip-install it in the container: it pulls CPU
onnxruntime and breaks GPU for FaceFusion).

Models (delivered by manifest):
- models/buffalo_l/{det_10g,genderage,w600k_r50}.onnx
- models/codeformer.onnx
- inswapper_128.onnx is reused from Facefusion_comfyui/models (fallback: models/).

Node: **PositionalRegionSwap** (image, plan JSON [{"sex","is_bj"}...] in region order,
restore_blend, ref_1..ref_3) -> swapped image.
