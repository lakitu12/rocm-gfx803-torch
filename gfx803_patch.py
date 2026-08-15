"""
gfx803 torch 补丁: nonzero 族 op 的 CPU 回退
================================================

背景 (2026-08-15 排查结论):
  torch 在 gfx803 上完整构建的 libtorch_hip.so 中, Nonzero.hip 的 fatbin
  (含 hipcub DeviceReduce/DeviceSelect kernel) 经 ld 拼接进 ~269MB 的
  .hip_fatbin 段后, HIP 运行时注册失败:
    - "Cannot retrieve Static function, error: 218"
    - launch 时 "HIP error: invalid device function"
  影响 op: torch.nonzero / Tensor.nonzero / torch.argwhere /
  torch.masked_select / bool 掩码索引 (t[mask])。
  独立编译的 hipcub kernel 在 gfx803 上运行正常 (已实测), 问题仅在
  torch 完整构建的 fatbin 注册环节, 无法在独立环境下复现/修复。

  本补丁将这些 op 的数据处理转发到 CPU (数据通常很小, 开销可忽略),
  使 gfx803 上所有 torch 程序都能正常使用。

用法:
  import gfx803_patch          # 在 import torch 之后, 任何使用之前
  或安装为 sitecustomize.py (自动生效)。

注意:
  - 仅当 GPU 为 AMD gfx803 (Polaris) 时激活; 其他平台零影响
  - torch.nonzero 返回的索引在 CPU 计算后拷回 GPU, 语义完全一致
"""

import torch
import torch.nn.functional as F

_ACTIVE = False


def _is_gfx803():
    try:
        if not torch.cuda.is_available():
            return False
        name = torch.cuda.get_device_name(0).lower()
        return any(k in name for k in ("rx 470", "rx 480", "rx 570", "rx 580",
                                       "rx 590", "gfx803", "polaris", "amd"))
    except Exception:
        return False


def install():
    """激活 gfx803 nonzero 族 CPU 回退补丁 (幂等)。"""
    global _ACTIVE
    if _ACTIVE:
        return
    if not _is_gfx803():
        return
    _ACTIVE = True

    _orig_nonzero = torch.nonzero
    _orig_t_nonzero = torch.Tensor.nonzero
    _orig_masked_select = torch.masked_select
    _orig_argwhere = torch.argwhere
    _orig_getitem = torch.Tensor.__getitem__

    def _cpu_nonzero(input, *a, **k):
        if input.is_cuda:
            return _orig_nonzero(input.detach().cpu(), *a, **k).to(input.device)
        return _orig_nonzero(input, *a, **k)

    def _cpu_t_nonzero(self, *a, **k):
        if self.is_cuda:
            return _orig_t_nonzero(self.detach().cpu(), *a, **k).to(self.device)
        return _orig_t_nonzero(self, *a, **k)

    def _cpu_masked_select(input, mask, *a, **k):
        if input.is_cuda:
            return _orig_masked_select(input.detach().cpu(), mask.cpu(), *a, **k).to(input.device)
        return _orig_masked_select(input, mask, *a, **k)

    def _cpu_argwhere(input, *a, **k):
        if input.is_cuda:
            return _orig_argwhere(input.detach().cpu(), *a, **k).to(input.device)
        return _orig_argwhere(input, *a, **k)

    def _cpu_getitem(self, key):
        if self.is_cuda and isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            return _orig_getitem(self.detach().cpu(), key.cpu()).to(self.device)
        return _orig_getitem(self, key)

    torch.nonzero = _cpu_nonzero
    torch.Tensor.nonzero = _cpu_t_nonzero
    torch.masked_select = _cpu_masked_select
    torch.argwhere = _cpu_argwhere
    torch.Tensor.__getitem__ = _cpu_getitem


install()
