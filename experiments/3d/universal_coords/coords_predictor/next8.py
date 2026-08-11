#*Thx*.nii or *Tho*.nii or *Abd*.nii as pattern



selected_input_path = "/nfs/data/nii/data1/jungm___ChemoTox/10116066/20220316122148/11_Thx_Abd_DE_venoes/Thx_Abd_DE_KM_3_0_Bf40_3_F_0_8_s002.nii"
model_file = "/nfs/data/nii/data1/Analysis/raua___BodyComp/ANALYSIS_coord/next8/model_patchwork.json"
out = "/home/dpxuser/dev/patch_icl/experiments/3d/universal_coords/coords_predictor/output/pred_next8_coords.nii.gz"
json_out = "/home/dpxuser/dev/patch_icl/experiments/3d/universal_coords/coords_predictor/output/pred_next8_info.json"

import sys
sys.path.append("/software")
import nibabel as nib
import patchwork2.model as patchwork
import patchwork2.improc_utils as ip
import tensorflow as tf
import json
import numpy as np
import os.path
from pathlib import Path


def grid_fingerprint(img):
    R = img.affine[:3, :3]
    spacing = np.linalg.norm(R, axis=0)
    direction = R / spacing
    N = np.array(img.shape[:3])
    return {
        "origin": img.affine[:3, 3],
        "direction": direction,
        "fov_edge": N * spacing,          # align_corners=False convention
        "fov_center": (N - 1) * spacing,  # align_corners=True convention
    }

def same_source_image(a, b, atol=1e-1):
    fa, fb = grid_fingerprint(a), grid_fingerprint(b)
    same_orient = (np.allclose(fa["origin"], fb["origin"], atol=atol)
                   and np.allclose(fa["direction"], fb["direction"], atol=1e-4))
    same_fov = (np.allclose(fa["fov_edge"],   fb["fov_edge"],   atol=atol)
                or np.allclose(fa["fov_center"], fb["fov_center"], atol=atol))
    return same_orient and same_fov

pid, sid = Path(selected_input_path).parts[6:8]

nii_files = DPX_selectFiles('jungm___ChemoTox',f'{pid}#{sid} **/*.nii',True)

i_max_file = np.argmax([os.path.getsize(f) for f in nii_files])

dicts = DPX_selectFiles('jungm___ChemoTox',f'{pid}#{sid} **/*.nii',False)

assert len(dicts) == 1

file_dicts = dicts[f"{pid}#{sid}"]

max_file_size = -np.inf
max_file = None
for filename in file_dicts:
    file_size = file_dicts[filename]["filesize"]
    if file_size > max_file_size:
        max_file_size = file_size
        max_file = filename

input_file = file_dicts[max_file]["FilePath"]

same_selected_file = input_file == selected_input_path
existing_out_nii = nib.load(out)
wanted_in_nii = nib.load(input_file)

same_spatial_locations = same_source_image(existing_out_nii, wanted_in_nii)

if not (same_spatial_locations and same_selected_file):
    print(f"Predicting coords for {input_file}")    
    if len(nib.load(f1).shape) > 3:
        crop_fdim = [0]
    else:
        crop_fdim = None
        model = patchwork.PatchWorkModel.load(f5)
        nii, res = model.apply_on_nifti(
            [input_file],
            out,
            generate_type="random",
            out_typ="float32",
            repetitions=30,
            num_chunks=10,
            branch_factor=2,
            input_transform=lambda x: tf.where(x<-1000.0,-1000.0,x),    
            postproc=lambda x: model.finalBlock.decodeCoords(x[..., 1:]),
            level="mixnohead",
            augment={},
            scale_to_original=False,
            crop_fdim=crop_fdim,
        )
else:
    print(f"Correct file {input_file} was already selected")

info_dict = dict(
    input_file_path=input_file,
    input_file_name=max_file
)


json.dump(info_dict, open(json_out, "w"))