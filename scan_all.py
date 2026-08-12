#!/usr/bin/env python3
"""全 shape 扫描: 每个 shape 打印 err + 选中的 solution index。
TENSILE_DB=0x2 输出到 stderr, 解析最后选中的 index。
用法: TENSILE_DB=0x2 python3 scan_all.py > errs.tsv 2> db.log
"""
import sys, os, torch, re
sys.path.insert(0, '/home/lakitu/.local/lib/python3.14/site-packages')

SHAPES = [
    ('linear', (1, 320, 320)), ('linear', (2, 320, 320)),
    ('linear', (64, 320, 320)), ('linear', (64, 320, 1280)),
    ('linear', (2, 320, 1280)), ('linear', (4096, 320, 320)),
    ('linear', (4096, 320, 1280)), ('linear', (1024, 320, 320)),
    ('linear', (2, 1024, 1024)), ('linear', (64, 1280, 1280)),
    ('linear', (2, 1280, 1280)), ('linear', (4096, 1280, 1280)),
    ('linear', (77, 768, 768)), ('linear', (2, 768, 768)),
    ('bmm_qk', (2, 8, 64, 40)), ('bmm_qk', (2, 8, 4096, 40)),
    ('bmm_qk', (1, 8, 4096, 64)), ('bmm_sv', (2, 8, 64, 40)),
    ('bmm_sv', (1, 8, 4096, 64)),
]

def run_one(kind, args):
    if kind == 'linear':
        N, K, M = args
        x = torch.randn(N, K, device='cuda', dtype=torch.float16)
        w = torch.randn(M, K, device='cuda', dtype=torch.float16)
        y = torch.nn.functional.linear(x, w)
        ref = torch.nn.functional.linear(x.float(), w.float())
        return (y.float() - ref).abs().max().item()
    elif kind == 'bmm_qk':
        b, h, s, d = args
        q = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
        k = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
        y = torch.matmul(q, k.transpose(-2, -1))
        ref = torch.matmul(q.float(), k.float().transpose(-2, -1))
        return (y.float() - ref).abs().max().item()
    elif kind == 'bmm_sv':
        b, h, s, d = args
        q = torch.randn(b, h, s, s, device='cuda', dtype=torch.float16)
        v = torch.randn(b, h, s, d, device='cuda', dtype=torch.float16)
        y = torch.matmul(q, v)
        ref = torch.matmul(q.float(), v.float())
        return (y.float() - ref).abs().max().item()

print("shape\tverdict\terr", flush=True)
for kind, args in SHAPES:
    try:
        err = run_one(kind, args)
        verdict = 'OK' if err < 0.1 else 'BAD'
    except Exception:
        err = float('inf'); verdict = 'EXC'
    print(f"{kind}{args}\t{verdict}\t{err:.4g}", flush=True)
