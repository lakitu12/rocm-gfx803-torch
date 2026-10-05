#!/usr/bin/env python3
# CI 门禁: 链接前逐 .o 扫 fatbin 设备码元数据 uses_dynamic_stack。
# 背景(run 37261601246 wheel 真机实证): clang23 -O0 给 TensorTopK 1377/1378 kernel
# 打上 dynamic_stack=true, gfx803 scratch 分配路径不可靠 -> topk launch 即
# Memory access fault on (nil), 整机 GPU hang。该缺陷纯主机侧可见, 不必炸机试错。
# 判据: 目标 fatbin 中 uses_dynamic_stack true 计数必须为 0;
#       另对照同树正常对象(如 AbsKernel.hip.o)必须存在且 false>0, 防"空扫假绿"。
import subprocess, struct, sys, os, re

CL = None
for c in ['/opt/rocm/lib/llvm/bin', '/opt/rocm-10.0.0/lib/llvm/bin']:
    if os.path.exists(c + '/llvm-objcopy'):
        CL = c; break
assert CL, 'llvm-objcopy not found'

def fatbin_of(obj):
    tmp = '/tmp/_gate_fb.bin'
    subprocess.run([f'{CL}/llvm-objcopy', '--dump-section=.hip_fatbin=' + tmp, obj, '/dev/null'],
                   capture_output=True)
    if not os.path.exists(tmp):
        return None
    d = open(tmp, 'rb').read()
    os.remove(tmp)
    return d

def dyn_stack_stats(d):
    # __CL 容器: msgpack 元数据明文; 键后跟值, true=0xc3? 实际 ROCm 元数据用 0x01/0x00 单字节
    pos = 0; t = 0; f = 0
    while True:
        pos = d.find(b'uses_dynamic_stack', pos + 1)
        if pos < 0: break
        nxt = d[pos + 18:pos + 21]
        # msgpack fixmap 后常见 [0xd9,0x13,key, 0x??]: bool true=0xc3 false=0xc2 或裸 01/00
        if len(nxt) >= 1:
            if nxt[0] in (0xc3, 0x01): t += 1
            elif nxt[0] in (0xc2, 0x00): f += 1
    return t, f

def main():
    objs = sys.argv[1:]
    assert objs, 'usage: fatbin_gate.py <torch_hip .o 文件...>'
    bad = 0; scanned = 0
    for o in objs:
        d = fatbin_of(o)
        if d is None: continue
        t, f = dyn_stack_stats(d)
        scanned += 1
        if t > 0:
            print(f'FAIL {o}: uses_dynamic_stack true={t} (false={f})')
            bad += 1
        elif f == 0:
            print(f'FAIL {o}: fatbin 无 kernel 元数据(空扫假绿?)')
            bad += 1
    # 假绿防护: 至少扫到 1 个含元数据的 fatbin
    if scanned == 0:
        print('FAIL: 没有任何对象含 .hip_fatbin 段(构建产物异常)')
        sys.exit(2)
    if bad:
        print(f'GATE FAIL: {bad} 个对象 dynamic_stack 异常 / 共扫 {scanned}')
        sys.exit(1)
    print(f'GATE OK: {scanned} 个 fatbin 全部 dynamic_stack=false')

main()
