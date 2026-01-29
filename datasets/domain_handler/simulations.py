# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple, Iterable, Sequence, Any
import numpy as np
import h5py

from ..utils import euler_to_rotate6d, quat_to_rotate6d
from .base import BaseHDF5Handler


# ------------------------------- Calvin --------------------------------------
class CalvinHandler(BaseHDF5Handler):
    """Calvin (sim): proprio [T,7] -> xyz(3)+euler_xyz(3)+grip(1). Right is zeros."""
    dataset_name = "Calvin"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        proprio = f["proprio"][()]  # [T,7]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:] < 0.],
            axis=-1,
        )  # [T,10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 20))


# --------------------------------- RT1 ---------------------------------------
class RT1Handler(BaseHDF5Handler):
    """RT1 (sim-like packaging): eef_quat_orientation [T,7], gripper [T,1]."""
    dataset_name = "RT1"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 3.0, 10.0
        eefq = f["eef_quat_orientation"][()]  # [T,7] pos3 + quat4
        grip = f["gripper"][()]               # [T,1] or [T]
        if grip.ndim == 1:
            grip = grip[:, None]
        left = np.concatenate([eefq[:, :3], quat_to_rotate6d(eefq[:, 3:]), grip], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 6))


# ------------------------------- Bridge --------------------------------------
class BridgeHandler(BaseHDF5Handler):
    """
    Bridge (sim). HDF5:
      /proprio [T, >=6] -> xyz(3) + euler_xyz(3) + ...
      /action  [T, ...] -> last channel is gripper (1=open), we convert to (1=closed)
    Output left/right: [T,10] = xyz(3)+rot6d(6)+grip(1). Single arm → right zeros.
    """
    dataset_name = "Bridge"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 5.0, 5.0
        proprio = f["proprio"][()]                     # [T, >=6]
        action  = f["action"][()]                      # [T, ...]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), 1 - action[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------- LIBERO --------------------------------------
# class BaseHDF5Handler(DomainHandler):
#     """
#     Generic HDF5 handler with resource-safe iteration.

#     Subclasses only implement:
#       - build_left_right(f) -> (left, right, left_time, right_time, freq, qdur)
#           left/right: abs_trajectory [T, C], left_time/right_time: optional time arrays [T],
#           freq (Hz), qdur (seconds of future window)
#       - index_candidates(T_left, training) -> Iterable[int]

#     Optionally override:
#       - get_image_datasets(f): sequence of image arrays/datasets
#       - read_instruction(f): string instruction
#     """

#     # --- Optional overrides -------------------------------------------------
#     def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
#         keys: Sequence[str] = self.meta["observation_key"]
#         return [f[k][()] for k in keys]

#     def read_instruction(self, f: h5py.File) -> str:
#         key: str = self.meta["language_instruction_key"]
#         ds = f[key]
#         v = ds[()]
#         return v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()

#     # --- Required hooks -----------------------------------------------------
#     def build_left_right(
#         self, f: h5py.File
#     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
#         raise NotImplementedError

#     def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
#         raise NotImplementedError
#     # -----------------------------------------------------------------------

#     @staticmethod
#     def _pil_from_arr(arr: Any) -> Image.Image:
#         from ..utils import decode_image_from_bytes
#         return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

#     def iter_episode(
#         self,
#         traj_idx: int,
#         *,
#         num_actions: int,
#         training: bool,
#         image_aug,
#         lang_aug_map: dict | None,
#         **kwargs
#     ) -> Iterable[dict]:
#         """Open once, yield many samples; file is always closed on exit."""
#         datapath = self.meta["datalist"][traj_idx]
#         if not isinstance(datapath, str):
#             datapath = datapath[0]
           
#         # # 修复：适配字典/列表/字符串类型的 datapath
#         # if isinstance(datapath, dict):
#         #     # 如果是字典，取第一个值（常见键名：'path'/'file'/'data_path'，可根据实际调整）
#         #     # 优先取 'path' 键，没有则取第一个值
#         #     datapath = datapath.get('path', next(iter(datapath.values())))
#         # elif isinstance(datapath, (list, tuple)) and len(datapath) > 0:
#         #     # 如果是列表/元组，取第一个元素（保留原有逻辑）
#         #     datapath = datapath[0]
#         # elif not isinstance(datapath, str):
#         #     # 其他非字符串类型，转为字符串
#         #     datapath = str(datapath)


#         with _open_h5(datapath) as f:
#             # Images and mask
#             images = self.get_image_datasets(f)
#             # Language
#             ins = self.read_instruction(f)
#             # Domain-specific kinematics and timing
#             left, right, lt, rt, freq, qdur = self.build_left_right(f)
        
        
#         image_mask = torch.zeros(self.num_views, dtype=torch.bool)
#         image_mask[:len(images)] = True
#         if lt is None: lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
#         if rt is None: rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

#         # Candidate indices (optionally shuffled)
#         idxs = list(self.index_candidates(left.shape[0], training))
#         if training: random.shuffle(idxs)

#         # Interpolators; clamp to endpoints
#         L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
#         R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
#         ref = (lt + rt) / 2.0

#         V = min(self.num_views, len(images))
#         for idx in idxs:

#             # Query future window
#             cur = ref[idx]
#             q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
#             lseq = torch.tensor(L(q))
#             rseq = torch.tensor(R(q))

#             # Skip static segments
#             if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5: continue
            
#             # Language augmentation
#             if training and lang_aug_map and ins in lang_aug_map:
#                 ins = random.choice(lang_aug_map[ins])
            
#             imgs = [image_aug(self._pil_from_arr(images[v][idx])) for v in range(V)]
#             while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
#             image_input = torch.stack(imgs, dim=0)

#             yield {
#                 "language_instruction": ins,
#                 "image_input": image_input,
#                 "image_mask": image_mask,
#                 "abs_trajectory": torch.cat([lseq, rseq], -1).float()
#             }


# handler = Handler(meta=meta, num_views=self.num_views)
class LiberoHandler(BaseHDF5Handler):
    """
    LIBERO (sim). HDF5:
      /abs_action_6d [T,10] = xyz(3)+rot6d(6)+grip_raw(1). Single arm.
    Also drops first frame for images (matches original pipeline behavior).
    """
    dataset_name = "libero"

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys = self.meta["observation_key"]
        images = [f[k] for k in keys]
        # Drop the first frame (image desync quirk in original data)
        return [img[1:] for img in images]

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        a = f["abs_action_6d"][()]                             # [T,10]
        left = np.concatenate([a[:, :9], (a[:, 9:] > 0.0)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------ VLABench -------------------------------------
class VLABenchHandler(BaseHDF5Handler):
    """
    VLABench (sim). HDF5:
      /proprio [T, >=7] -> xyz(3) + euler_xyz(3) + grip(1).
    Single arm → right zeros.
    """
    dataset_name = "VLABench"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        proprio = f["proprio"][()]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 15))


# ------------------------------ RobotWin2 ------------------------------------
class RobotWin2Handler(BaseHDF5Handler):
    """
    robotwin2_abs_ee / robotwin2_clean (sim). HDF5:
      /endpose/left_endpose   [T,7]  xyz(3)+quat(4)
      /endpose/right_endpose  [T,7]
      /endpose/left_gripper   [T]    1=open  -> convert to 1=closed
      /endpose/right_gripper  [T]
    Output both arms. freq≈30Hz, qdur=1s.
    """
    dataset_name = "robotwin2-*"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        l = f["endpose/left_endpose"][()]                      # [T,7]
        r = f["endpose/right_endpose"][()]                     # [T,7]
        lg = (1 - f["endpose/left_gripper"][()][:, None])      # [T,1] 1=closed
        rg = (1 - f["endpose/right_gripper"][()][:, None])
        left  = np.concatenate([l[:, :3], quat_to_rotate6d(l[:, 3:]), lg], axis=-1)
        right = np.concatenate([r[:, :3], quat_to_rotate6d(r[:, 3:]), rg], axis=-1)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ---------------------------- Robocasa-Human ---------------------------------
class RobocasaHumanHandler(BaseHDF5Handler):
    """
    robocasa-human (teleop in sim). HDF5:
      /action_dict/abs_pos     [T,3]
      /action_dict/abs_rot_6d  [T,6]
      /action_dict/gripper     [T,1]  ( >0 => closed )
    Single arm → right zeros.
    """
    dataset_name = "robocasa-human"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        left = np.concatenate(
            [
                f["action_dict/abs_pos"][()],
                f["action_dict/abs_rot_6d"][()],
                (f["action_dict/gripper"][()] > 0.0).astype(np.float32),
            ],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))
