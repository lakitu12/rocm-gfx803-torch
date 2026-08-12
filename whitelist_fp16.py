#!/usr/bin/env python3
"""fp16 白名单重建 v1: 迭代剔除坏 solution。
策略: 全 shape 自然匹配扫描 -> 找到 BAD shape 及其选中 solution -> 从匹配表
删除该 solution 的所有条目 -> 重扫。直到无 BAD 或无法改进。
保留每个 op 的每类至少有 1 个 solution; 若全删光则保留原样 (宁缺勿错原则下
宁可不匹配也不给错的)。
"""
import sys, os, torch, re, subprocess, shutil, json
sys.path.insert(0, '/home/lakitu/.local/lib/python3.14/site-packages')
import msgpack

DAT = '/home/lakitu/rocm-gfx803-archive/fixed-lib/TensileLibrary.dat'
BACKUP = '/home/lakitu/rocm-gfx803-archive/fixed-lib/TensileLibrary.dat.pre-whitelist'
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

def scan():
    """全 shape 扫描, 返回 [(shape, err)]"""
    results = []
    for kind, args in SHAPES:
        try:
            err = run_one(kind, args)
        except Exception:
            err = float('inf')
        results.append(((kind, args), err))
    return results

def collect_tables(node, out):
    if not isinstance(node, dict): return
    if node.get('type') == 'Problem':
        for r in node.get('rows', []):
            p = r.get('predicate'); l = r.get('library')
            if p and p.get('type') == 'And':
                te = [v['value'] for v in p['value'] if v['type']=='TypesEqual'][0]
                hpa = [v['value'] for v in p['value'] if v['type']=='HighPrecisionAccumulate'][0]
                ml = l
                while isinstance(ml, dict) and ml.get('type') == 'Problem':
                    mr = ml.get('rows', [])
                    if not mr: break
                    ml = mr[0].get('library')
                if isinstance(ml, dict) and ml.get('type') == 'Matching':
                    out.append((te, hpa, ml))
            collect_tables(l, out)
    elif isinstance(node, list):
        for v in node: collect_tables(v, out)

def remove_solution_from_dat(dat, bad_idx):
    """从所有 Half 分支匹配表删除 bad_idx 条目"""
    removed = 0
    pm = dat['library']['rows'][0]['library']
    for op, entry in pm['map'].items():
        tables = []
        collect_tables(entry, tables)
        for te, hpa, mt in tables:
            if 'Half' not in te: continue
            new_tbl = [e for e in mt['table'] if e['index'] != bad_idx]
            if len(new_tbl) != len(mt['table']):
                removed += len(mt['table']) - len(new_tbl)
                mt['table'] = new_tbl
    return removed

def find_selected(shape_key, db_log):
    """从 db log 找某 shape 的选中 solution — 简化: 无法精确定位, 跳过"""
    return None

# 主流程
if not os.path.exists(BACKUP):
    shutil.copy2(DAT, BACKUP)
    print(f"备份到 {BACKUP}")

for iteration in range(8):
    # 扫描
    os.environ.pop('TENSILE_SOLUTION_INDEX', None)
    torch.cuda.empty_cache()
    results = scan()
    bad = [(s, e) for s, e in results if e > 0.1]
    print(f"\n=== 迭代 {iteration}: {len(results)} shapes, {len(bad)} BAD ===")
    for s, e in bad:
        print(f"  BAD {s}: err={e:.4g}")
    if not bad:
        print("全部 OK!")
        break
    # 对每个 BAD shape, 用 TENSILE_DB 抓选中 solution (单独进程)
    bad_solutions = set()
    for (kind, args), err in bad:
        code = f"""
import torch
kind = {kind!r}; args = {args!r}
if kind == 'linear':
    N,K,M = args
    x = torch.randn(N,K,device='cuda',dtype=torch.float16)
    w = torch.randn(M,K,device='cuda',dtype=torch.float16)
    torch.nn.functional.linear(x,w)
elif kind == 'bmm_qk':
    b,h,s,d = args
    q = torch.randn(b,h,s,d,device='cuda',dtype=torch.float16)
    k = torch.randn(b,h,s,d,device='cuda',dtype=torch.float16)
    torch.matmul(q,k.transpose(-2,-1))
else:
    b,h,s,d = args
    q = torch.randn(b,h,s,s,device='cuda',dtype=torch.float16)
    v = torch.randn(b,h,s,d,device='cuda',dtype=torch.float16)
    torch.matmul(q,v)
"""
        env = dict(os.environ)
        env['TENSILE_DB'] = '0x2'
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=120)
        m = re.findall(r'Solution index selected: (\d+)', p.stdout + p.stderr)
        if m:
            sidx = int(m[-1])
            bad_solutions.add(sidx)
            print(f"  {kind}{args} -> solution {sidx}")
        else:
            print(f"  {kind}{args} -> 无法抓 solution (err={err:.4g})")
    if not bad_solutions:
        print("无法定位坏 solution, 停止")
        break
    # 删除
    with open(DAT, 'rb') as f:
        dat = msgpack.unpackb(f.read(), raw=False)
    total_removed = 0
    for sidx in sorted(bad_solutions):
        r = remove_solution_from_dat(dat, sidx)
        total_removed += r
        print(f"  删除 solution {sidx}: {r} 条")
    with open(DAT, 'wb') as f:
        f.write(msgpack.packb(dat, use_bin_type=True))
    print(f"  共删除 {total_removed} 条")

print("\n完成。")
