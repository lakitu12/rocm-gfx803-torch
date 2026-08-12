# rocm-gfx803-torch

gfx803 (AMD Polaris: RX 580 / RX 590 GME) 的 ROCm 6.4.3 工具链 + 自编译 torch。

## 目的

1. **备份**: 本地定制版 ROCm 6.4.3 (带 gfx803 支持) 完整备份, 防丢
2. **CI 重编 torch**: GitHub Actions 上编译修复 wave64 LayerNorm bug 的 torch,
   本地零负担, wheel 下载安装即可

## Release 资产

### `rocm-6.4.3` (3.84GB, 2 分卷)

本地 `/opt/rocm-6.4.3` 的完整 zstd 打包 (14GB → 3.84GB)。

```bash
# 下载后合并解压:
gh release download rocm-6.4.3 --pattern 'rocm-6.4.3.part.*'
cat rocm-6.4.3.part.* > rocm-6.4.3.tar.zst
sudo tar --zstd -xf rocm-6.4.3.tar.zst -C /opt   # 解压出 /opt/rocm-6.4.3
```

## GitHub Actions

### `build-torch-gfx803` — 编译 torch (LayerNorm wave64 修复版)

- checkout pytorch `8d1791e` (2.8.0)
- patch `c10/macros/Macros.h`: `__GFX9__` → `__GFX7__||__GFX8__||__GFX9__`
  (GCN 全系 wave64; 原版只认 GFX9, 导致 gfx803 的 warp 归约丢 lane 32-63 数据,
  LayerNorm 数值全错)
- `PYTORCH_ROCM_ARCH=gfx803 python setup.py bdist_wheel`
- 产物: `torch-2.8.0-gfx803` artifact

## 本地环境速查

```bash
source ~/rocm-gfx803-archive/rocblas-env.sh   # LD_LIBRARY_PATH + ROCBLAS_TENSILE_LIBPATH + HSA_OVERRIDE
```

详见 `ROCM-RESEARCH.md` (完整研究存档, 含 fp16 白名单/fixed-lib/LayerNorm 修复全过程)。
