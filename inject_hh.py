#!/usr/bin/env python3
"""把系统版 rocBLAS 4.4.1 的 HH/HH_HPA (fp16) 分支注入 fixed-lib 的 TensileLibrary.dat。

背景: fixed-lib 重构时只保留了 fp32 分支的 Matching 表, fp16 (Half) 分支被清空,
      导致 torch 的 fp16 matmul/linear 走 rocBLAS gemm_ex 时报
      "No Tensile solution found"。系统版 4.4.1 split dat 有完整 HH gfx803 分支。

机制: rocBLAS 启动时 glob 预加载目录内所有 *gfx803*co 文件, launch 时按
      kernel symbol 名 (= solution name) 匹配, 与 dat 里的 codeObjectFile 无关。

步骤:
1. 读系统版 8 个 HH/HH_HPA fallback.dat (4 布局 x 2 HPA)
2. solutions 追加到 fixed-lib, index 重映射 (从 927 起, 原 index 冲突)
3. 树中每个 op 添加 Half 分支: And[TypesEqual[Half,Half,Half,Half], HPA, F32XdlMathOp=Float, SR=False] -> Problem -> TruePred -> Matching
4. 复制系统版 gfx803 hsaco 到 fixed-lib 目录
"""
import sys, shutil, glob, os
sys.path.insert(0, '/home/lakitu/.local/lib/python3.14/site-packages')
import msgpack, json, copy

FIXED = '/home/lakitu/rocm-gfx803-archive/fixed-lib/TensileLibrary.dat'
SYS   = '/usr/lib/x86_64-linux-gnu/rocblas/4.4.1/library'
DST   = '/home/lakitu/rocm-gfx803-archive/fixed-lib'

OPS = ['Contraction_l_Alik_Bljk_Cijk_Dijk', 'Contraction_l_Ailk_Bljk_Cijk_Dijk',
       'Contraction_l_Ailk_Bjlk_Cijk_Dijk', 'Contraction_l_Alik_Bjlk_Cijk_Dijk']
TYPES = ['HH', 'HH_HPA']  # HH: HPA=False; HH_HPA: HPA=True (fp16 输入 fp32 累积)

with open(FIXED, 'rb') as f:
    fixed = msgpack.unpackb(f.read(), raw=False)

sols = fixed['solutions']
next_idx = max(s['index'] for s in sols) + 1
print(f"fixed-lib solutions: {len(sols)}, next free index: {next_idx}")

added = 0
for t in TYPES:
    for op in OPS:
        sysf = f'{SYS}/TensileLibrary_Type_{t}_{op}_fallback.dat'
        if not os.path.exists(sysf):
            print(f"  MISSING {sysf}"); continue
        with open(sysf, 'rb') as f:
            sd = msgpack.unpackb(f.read(), raw=False)
        sys_sols = sd['solutions']
        # 复制 solutions 并重映射 index
        idx_map = {}
        for s in sys_sols:
            ns = copy.deepcopy(s)
            ns['index'] = next_idx
            idx_map[s['index']] = next_idx
            sols.append(ns)
            next_idx += 1
        # 系统版 library: Problem -> rows[0].library = Matching
        match = sd['library']['rows'][0]['library']
        # 重映射 matching 表的 index
        tbl = []
        for e in match['table']:
            ne = copy.deepcopy(e)
            ne['index'] = idx_map[ne['index']]
            tbl.append(ne)
        # 构造 Half 分支: And(...) -> Problem -> [TruePred -> Matching]
        hpa = (t == 'HH_HPA')
        and_pred = {'type': 'And', 'value': [
            {'type': 'TypesEqual', 'value': ['Half', 'Half', 'Half', 'Half']},
            {'type': 'HighPrecisionAccumulate', 'value': hpa},
            {'type': 'F32XdlMathOp', 'value': 'Float'},
            {'type': 'StochasticRounding', 'value': False}]}
        branch = {'predicate': and_pred, 'library': {
            'type': 'Problem', 'rows': [{'predicate': {'type': 'TruePred'},
                                         'library': {'type': 'Matching',
                                                     'properties': match['properties'],
                                                     'table': tbl,
                                                     'distance': match['distance']}}]}}
        # 找到 fixed-lib 树中该 op 的 Problem(存 And 分支的那层)
        pm = fixed['library']['rows'][0]['library']  # gfx803 Hardware -> ProblemMap
        entry = pm['map'][op]
        # 树: Problem -> rows[0] -> TruePred -> Problem(And 分支层) -> rows
        ands_problem = entry['rows'][0]['library']['rows'][0]['library']
        assert ands_problem['type'] == 'Problem', f"unexpected {ands_problem['type']}"
        # 追加 Half 分支
        ands_problem['rows'].append(branch)
        added += len(sys_sols)
        print(f"  + {t} {op}: {len(sys_sols)} sols, {len(tbl)} match keys -> branch appended")

# 写回
with open(FIXED, 'wb') as f:
    f.write(msgpack.packb(fixed, use_bin_type=True))
print(f"\nTotal solutions now: {len(sols)}, added {added}")

# 复制系统版 gfx803 hsaco (kernel symbol 名 = solution name, rocBLAS 按名匹配)
copied = 0
for t in TYPES:
    for op in OPS:
        src = f'{SYS}/TensileLibrary_Type_{t}_{op}_fallback_gfx803.hsaco'
        if os.path.exists(src):
            shutil.copy2(src, DST)
            copied += 1
print(f"copied {copied} hsaco files")
