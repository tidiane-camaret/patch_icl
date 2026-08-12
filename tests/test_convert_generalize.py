import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.convert_to_npy import _convert_chemotox


def _write_nii(path, arr, affine):
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def test_chemotox_labels_resampled_onto_ct_grid(tmp_path):
    """The CT and its masks may live on different native grids (shape/spacing/origin).
    The converter must resample every label onto the CT's output grid so ct.npy and
    label.npy/bc.npy come out the same shape AND world-aligned."""
    # CT: 20^3 @ 1.0mm, origin at 0.
    img = (np.random.rand(20, 20, 20) * 100 - 50).astype(np.float32)
    img_aff = np.diag([1.0, 1.0, 1.0, 1.0])
    # totalseg on a DIFFERENT grid: finer 0.5mm, larger extent, shifted origin. A cube of
    # TS-liver (id 5) at world [5,10]mm on every axis.
    ts = np.zeros((40, 40, 40), dtype=np.int16)
    ts_aff = np.diag([0.5, 0.5, 0.5, 1.0]); ts_aff[:3, 3] = [-5.0, -5.0, -5.0]
    ts[np.ix_(*[range(20, 30)] * 3)] = 5   # world 5..10mm (native idx = (w+5)/0.5)
    # bclabels 4D on yet another (coarser) grid.
    bc = np.zeros((10, 10, 10, 2), dtype=np.int16)
    bc_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    bc[..., 0][3:6, 3:6, 3:6] = 1          # muscle, world 6..12mm

    _write_nii(tmp_path / "img.nii", img, img_aff)
    _write_nii(tmp_path / "ts.nii", ts, ts_aff)
    _write_nii(tmp_path / "bc.nii", bc, bc_aff)

    out_dir = tmp_path / "out"
    task = {"source": "chemotox", "subj_id": "s0",
            "out_dir": str(out_dir), "overwrite": True, "size": None,
            "target_spacing": 1.0, "modality": "ct", "store_raw": False,
            "inputs": {"img": str(tmp_path / "img.nii"),
                       "totalseg": str(tmp_path / "ts.nii"),
                       "bclabels": str(tmp_path / "bc.nii")}}
    sid, status, sp, shape, _ = _convert_chemotox(task)
    assert status == "ok", status

    ct = np.load(out_dir / "ct.npy")
    lab = np.load(out_dir / "label.npy")
    bcn = np.load(out_dir / "bc.npy")
    # Alignment invariant the dataloader crop path depends on.
    assert ct.shape == lab.shape == bcn.shape
    # Labels landed (world-space resample kept the foreground, not dropped it).
    assert lab.max() > 0 and bcn.max() == 1
    # And they sit at the right world location: TS cube spanned world 5..10mm; at the CT's
    # 1mm/origin-0 grid that is voxels ~5..10.
    zs = np.array(np.where(lab > 0))
    assert zs.min() >= 3 and zs.max() <= 12
