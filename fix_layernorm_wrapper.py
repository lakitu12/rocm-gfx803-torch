#!/usr/bin/env python3
"""gfx803 LayerNorm 修复: torch 原生 layer_norm kernel 在 gfx803 上数值错误,
用 mean/var 原语手动实现 (已实测 1e-6 正确, 原生 2.56 错误)。
用法: python fix_layernorm_wrapper.py <original_main_args...>
"""
import sys
import torch
import torch.nn.functional as F

_orig_layer_norm = F.layer_norm
_orig_torch_layer_norm = getattr(torch, 'layer_norm', None)

def gfx803_layer_norm(*args, **kwargs):
    """基于 mean/var 原语的手动 LayerNorm (gfx803 上正确)。兼容任意调用形态。"""
    # 解析 (input, normalized_shape, weight, bias, eps) 位置/关键字
    if len(args) >= 5:
        input_, normalized_shape, weight, bias, eps = args[0], args[1], args[2], args[3], args[4]
    elif len(args) == 4:
        input_, normalized_shape, weight, bias = args
        eps = kwargs.get('eps', 1e-5)
    elif len(args) == 3:
        input_, normalized_shape, weight = args
        bias = kwargs.get('bias')
        eps = kwargs.get('eps', 1e-5)
    elif len(args) == 2:
        input_, normalized_shape = args
        weight = kwargs.get('weight')
        bias = kwargs.get('bias')
        eps = kwargs.get('eps', 1e-5)
    else:
        input_ = kwargs['input']
        normalized_shape = kwargs['normalized_shape']
        weight = kwargs.get('weight')
        bias = kwargs.get('bias')
        eps = kwargs.get('eps', 1e-5)

    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    if input_.device.type == 'cuda':
        ndim = input_.dim()
        norm_ndim = len(normalized_shape)
        reduce_dims = tuple(range(ndim - norm_ndim, ndim))
        mean = input_.mean(dim=reduce_dims, keepdim=True)
        var = input_.var(dim=reduce_dims, unbiased=False, keepdim=True)
        out = (input_ - mean) * torch.rsqrt(var + eps)
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        return out
    # CPU 分支: 调原始实现 (原始 F.layer_norm -> torch.layer_norm, 已保存)
    if _orig_torch_layer_norm is not None:
        return _orig_torch_layer_norm(input_, normalized_shape, weight, bias, eps)
    return _orig_layer_norm(input_, normalized_shape, weight, bias, eps)

F.layer_norm = gfx803_layer_norm
import torch.nn.functional as _F
_F.layer_norm = gfx803_layer_norm
if hasattr(torch, 'layer_norm'):
    torch.layer_norm = gfx803_layer_norm

print("[fix_layernorm] gfx803 LayerNorm patch applied (manual mean/var impl)", flush=True)

# 运行原 main.py
import runpy, os
sys.argv = ['main.py'] + sys.argv[1:]
main_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.py')
sys.path.insert(0, os.path.dirname(main_path))
os.chdir(os.path.dirname(main_path))
runpy.run_path(main_path, run_name='__main__')
