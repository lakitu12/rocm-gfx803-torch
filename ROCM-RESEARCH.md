# ROCm gfx803（RX 590 GME / Polaris）研究存档

> 2026-08 实测结论 + 测试/生成工具。本机：RX 590 GME（gfx803, 8GB VRAM）、ROCm 6.4.3 定制版、自编译 torch 2.8（~/ComfyUI/venv_gfx803）、Debian 14。

## 最终结论（全部实测）

| 项目 | 状态 |
|---|---|
| 硬件/驱动 | ✅ 正常（elementwise/归约/小矩阵/fp16 全精确） |
| rocBLAS（Tensile kernel）gfx803 | ❌ **全面不可用**：fp32 sgemm 错（512 错 25%、1024 错 99.5%）、fp64 dgemm 段错误、6.4 移除 hgemm/gemm_ex |
| 官方 ROCm 5.7 | ❌ 更差（fp32 512 就错 + torch wheel 无 gfx803 内核（1/387）） |
| Tensile master 4.44.0 重编 | ❌ bit 级同错（生成器深层 bug，LVCB=16/GLVWB=1 变体均失败） |
| hipBLASLt（torch 路径）fp16/bf16 | ✅ **全对**（transformer 端到端 0.09%） |
| hipBLASLt fp32 大 M（≥768） | ❌ 错（周期 4：C[i][j]=C[i][j+4]，VW4/线程映射类 bug） |
| **fp32 转置重组**（A@B→(Bᵀ@Aᵀ)ᵀ） | ✅ **任意尺寸对**（1e-4，应用层修复） |
| ZLUDA 自建 gfx803 | ✅ 编译链通（wave64 修复）+ 内核实测执行正确，但官方 torch wheel 纯 SASS 无 PTX 跑不了 |

**可用路线**：fp16/bf16 推理（可靠）；fp32 2D GEMM 用转置重组。

## 目录结构

```
tests/           C 测试（gemm_test.c 参数化 opA/opB/N；gemm_d.c fp64；gemm_pattern.c 坏元素模式；rocblas_test.c 基础验证）
                 gemm643/gemm_d 编译好的二进制（链接 6.4.3 rocBLAS，LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib 运行）
gen/             Tensile kernel 库生成流程（tensile_gen.sh + tcl_wrapper.py）
kernel-lib/      tensile_out/ 完整生成产物（TensileLibrary.dat 1.66MB + 637 hsaco + build_tmp 汇编源）
tensile-master.patch  Tensile master 的调试补丁（见下）
```

## 关键路径（持久）

- `~/Tensile-master` — Tensile 4.44.0（= rocBLAS 6.4 同版），含调试 patch
- `~/rocBLAS-src` — rocBLAS master（yaml 配置在 library/src/blas3/Tensile/Logic/asm_full/r9nano/）
- `~/dl/` — 下载证据：torch_221_rocm57.whl（官方 5.7 wheel）、rocm57/（5.7 运行时解包）、tf_fp16_test.py 等
- `~/zluda/built-gfx803` + `~/zluda-src` — ZLUDA gfx803 版

## Tensile kernel 库重建方法（30 分钟）

```bash
# 依赖：Tensile-master（含 patch）+ rocBLAS-src + ROCm 6.4.3 LLVM 工具链
# 运行：bash gen/tensile_gen.sh（已配好全部参数，输出 /tmp/tensile_out）
# 测试：LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib ROCBLAS_TENSILE_LIBPATH=/tmp/tensile_out/library ./tests/gemm643 0 0 1024
```

### 环境坑（都踩过，勿重踩）

1. **后台进程缺用户 site-packages**：Hermes 后台任务的环境没有 ~/.local —— 脚本里必须 `export PYTHONPATH="$PYTHONPATH:/home/lakitu/.local/lib/python3.14/site-packages"`（否则 import joblib/msgpack 失败，生成全程串行 + 打包 NameError）
2. **ld.lld 的 libxml2 警告**（"no version information"）污染 tryAssembler 输出 → gfx803 被误判不支持（已 patch Common.py 过滤）
3. **LibraryIO.writeMsgPack 的 msgpack 未导入**（已 patch 函数内 import；wrapper 也注入 LIO.msgpack）
4. **--generate-sources-and-exit 时汇编源**在 `/tmp/tensile_out/build_tmp/TENSILE_OUT/assembly/*.s`（带 gro 注释，调试地址计算用）
5. AsmCaps 检测通过后生成 = 789 任务（8 线程 ~30s）+ 编译 600+ kernel（8 线程 ~3min）

## Tensile-master patch 内容（tensile-master.patch）

1. `Common.py tryAssembler`：过滤 "no version information" 行
2. `LibraryIO.py writeMsgPack`：函数内 `import msgpack`
3. `TensileCreateLibrary.py processKernelSource`：dump 汇编 kernel 源到 /tmp/kernels_src/（调试用）
4. （SolutionStructs.py 的 LVCB 实验已还原；rocBLAS-src yaml 的 GLVWB 实验已还原）

## 变体试错记录（勿重复）

| 尝试 | 结果 |
|---|---|
| LVCB=16（SolutionStructs） | ❌ k 越界，错得更离谱 |
| GLVWB=1（yaml 强制 VW1） | ❌ 与原始完全一致 |
| 转置重组（应用层） | ✅ 唯一成功 |

## 坏 kernel 画像（rocprof 实抓）

- 512³：`Cijk_Ailk_Bljk_SB_MT32x8x32_..._GSU4_..._VW2`（错 25%）
- 1024³：`Cijk_Ailk_Bljk_SB_MT64x64x16_..._GSU1_..._VW4`（错 99.5%，周期 4）
- fp16/bf16（hipBLASLt）：全对；fp32 转置（opA=T 路径）：全对

## 诊断命令速查

```bash
# GEMM 数值测试（对比 CPU）
LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib ./tests/gemm643 0 0 1024   # opA opB N

# 抓实际执行的 kernel
LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib /opt/rocm-6.4.3/bin/rocprof -o /tmp/rp.csv ./tests/gemm643 0 0 1024

# 反汇编 kernel（地址计算分析）
/opt/rocm-6.4.3/lib/llvm/bin/llvm-objdump -d <hsaco> | grep -E 'v_add|v_lshl'

# torch 端到端 fp16 验证
~/ComfyUI/venv_gfx803/bin/python ~/dl/tf_fp16_test.py
```

## 2026-08-12 更新: fp32 GEMM 部分修复成功 (fixed-lib/)

**重大修正**: 之前的"rocBLAS fp32 全面不可用"结论部分错误 — 测试工具 (gemm_multi) 编译时
缺少 `--offload-arch=gfx803`, kernel 从未执行 (launch err=98 invalid device function),
所有 FAIL/PASS 判定都是假象 (naive 全 0 vs rocBLAS 输出对比)。

**修复方案** (实测驱动, 全部 bit 精确验证):
- `fixed-lib/` — 重构的 TensileLibrary.dat + 全部 code objects (966 文件, 22MB)
- 使用: `export ROCBLAS_TENSILE_LIBPATH=~/rocm-gfx803-archive/fixed-lib`

| 场景 | torch 的 problem | 分配 kernel | 结果 |
|---|---|---|---|
| a@b (matmul, V@attn) | Ailk_Bljk | **829** (MT32x8x32_GSU4, .co) | ✅ 64~8192+ 全尺寸 bit 精确 |
| a@b.t() (QK^T) | Alik_Bljk | **465** (MT128x64x12, .hsaco) | ✅ 全尺寸 bit 精确 |
| x@W.t() / nn.Linear | Ailk_Bjlk (transB=T) | 无好 kernel (802 等全错) | ❌ 无解, 系统库也 97% 错 |
| fp16/bf16 推理 | hipBLASLt | (不受影响) | ✅ 一直可用 |

**关键技术结论**:
1. **torch 路径与直接调 rocblas_sgemm 的 kernel 行为不同** (kernargs 差异) — 直接调用
   的测试结果不能代表 torch 场景; 验证必须用 torch 本身
2. torch 的 matmul 变换: a@b.t() → transA=T (Alik_Bljk); a.t()@b → transB=T (Ailk_Bjlk);
   nn.Linear → transA=T (M=out, N=batch)
3. 库重构方法: Matching 表按实测 PASS kernel 重写 (key=[n,n,n] 32 步进网格), 其他 fp32
   分支清空 (宁缺勿错), 非 fp32 分支原样保留
4. 偶发 HIPBLAS_STATUS_NOT_SUPPORTED: torch 后端选择有随机性 (hipBLASLt/rocBLAS 切换),
   重试/同进程多次调用可稳定

**构建工具链** (可复用):
- gemm_multi.c: 编译必须 `--offload-arch=gfx803` (否则 kernel invalid!)
- 测试判定: naive_gemm (GPU) 参考 — naive 本身也要 gfx803 编译
- 库重构脚本: /tmp/rebuild_grid.py 模式 (prune 树, Matching 表重写, 保留 distance 字段)
- TENSILE_DB=0x2 抓 "Solution index selected" (printPropertyEvaluation 位)
- rocBLAS 的 library 树: Hardware→ProblemMap(op)→Problem(predicate)→Matching(key→index)

### 2026-08-12 补充: Ailk_Bjlk (transB=T) 布局系统性全坏 — 重编验证

**结论**: Bjlk 布局 (B 转置) 在 gfx803 上 43/43 全坏, 与配置无关:
- 全部 42 个 solution (含 33 个无 key 的, MT8x8~MT128x128/VW1~VW4/source+assembly) 实测全错
- **重编 829 同款配置** (MT32x8x32/GSU4/GLVWA2/GLVWB1/VW2, Bjlk 布局改写, 生成器输出
  Cijk_Ailk_Bjlk_SB_MT32x8x32_SN_K1_gfx803.co) → 同样全错 (bad≈25万/512³)
- 根因: Tensile 生成器的 Bjlk (B 转置) 布局地址计算 bug, 与 tile/VW/source-asm 无关

**单配置 yaml 重编方法** (只生成 1 个 kernel, ~1 分钟):
1. yaml 顶层 list: [0]ver [1]schedule [2]arch [3]device [4]问题定义 [5]配置列表 [6]indexOrder [7]matching table
2. 配置块必须: `ProblemType: derive` (否则 solution 布局与文件名不符 → exit 255)
3. `AssignedProblemIndependentDerivedParameters: False` (重新派生)
4. data[7] 置空 [] (否则 index 越界); 生成后 dat 的 Matching 表为空 → 需自行注入 key 行
5. 布局改写: TransposeB/IndexAssignmentsB/IndexUnrollB/TLUB 从目标布局的块抄

**应用层兜底 (验证可用)**:
- `a.t()@b` 直接写会走 transB=T (坏路径, torch 的 view 转置处理把它转回 Bjlk)
- **物理转置**: `a.t().contiguous() @ b` → transA=N/transB=N (Ailk_Bljk→829) → ✅ bit 精确
- nn.Linear 场景: 权重预转置缓存 (W.t().contiguous()) 后 x@W 走 829 ✓

### 2026-08-12 补充: Bjlk 坏 kernel 汇编 diff 定位 (生成器模板 bug 实锤)

**对照材料**: `bjlk-diff/` 目录:
- `829_Ailk_Bljk.s` — 829 (Ailk_Bljk, 对, bit 精确) 的汇编源
- `Cijk_Ailk_Bjlk_SB_MT32x8x32_SN_K1.s` — 同配置 Bjlk 版 (错) 的汇编源

两者同 tile (MT32x8x32/GSU4/GLVWA2/GLVWB1/VW2)、同编译器 (ROCm 6.4.3 LLVM 18),
只差布局 (TransposeB/IndexAssignmentsB)。~2480 行几乎一致, 全部差异:

| 项目 | 829 (对) | Bjlk (错) |
|---|---|---|
| serial 位分配 (B) | groB-tile = serial/32 (高位), unroll = serial%32 | groB-tile = serial%8 (**低位**), unroll = serial/8 |
| LVCB | 32 | 8 |
| B 的 LDS 行距 | 0xa (MT1J+PAD=8+2) | 0x8 (无 pad) |
| GLOBAL_OFFSET_B | addr = OffsetL + StrideB1J*Offset1J | addr = Offset1J + StrideBL*OffsetL (转置适配, 公式正确) |
| local read (LDS→reg) | v3=(serial/16)%4, VW2 | **与 829 相同** |

**结论 (证据分级)**:
- [实测] 同 GPU 上 829 全尺寸 bit 精确 + fp16 全对 → **排除硬件**
- [推断→实测] 差异在生成器输出的程序结构 (serial 位分配/LVCB/PAD 是 Tensile 派生参数),
  非编译结果 → **LLVM codegen bug 基本排除**
- [推断] 错位藏在 LVCB=8 的 serial 位分配 × LDS 无 padding 布局的组合 — Bjlk 布局的
  LSCB/LVCB/PAD 派生参数 (SolutionStructs.py 的 roundupRatio 等) 在 gfx803 组合下不自洽;
  全局加载公式/列覆盖/行覆盖逐层验证均表面自洽 → 静态分析到此收敛极限

**将来修 Tensile 的起点**: 对比 LSCB/LVCB 派生 (`SolutionStructs.py` 的
`LVCB = roundupRatio(LSCB, GlobalLoadVectorWidthB)`) 在 Bljk vs Bjlk 布局的取值;
serial 分解模板 (KernelWriterAssembly.py) 对 LVCB 的位分配。注意 Bljk 布局还有部分配置坏
(866 MT64x64x16 等), 修 Bjlk 不等于全修。

## 2026-08-12 补充(下午): 重启后 GPU 计算栈整体退化 (环境级故障)

**12:31 重启(修桌面,11:04-12:31 间共 3 次重启)后,同一软件栈 fp32 全坏**:
- 系统库 + fixed-lib(829/465)全尺寸全错(max_err 48~73,确定性),kernel 确实执行(init_test: -777 全被覆盖)
- 上午验证(08:24-10:56, boot -3,无桌面)在同一份文件上 bit 精确 → **软件未变,GPU/驱动运行时状态变了**
- 显示激活非原因(sddm 停掉后照坏);内核/命令行/时钟/温度(49°C)/PCIe atomic(flags=1)/dmesg 全部正常

**可复现的症状矩阵**(测试工具都在 tests/ 下,已补全):
| 测试 | 结果 |
|---|---|
| naive_gemm(无 LDS) | ✅ 3.4e-5 |
| elementwise/归约(torch) | ✅ |
| fp16 hipBLASLt mm | ✅ 0.031(正常 fp16 累积误差) |
| 普通 LDS b32/b64/b128,1~32KB | ✅ 全对(lds_sizes/lds_width) |
| **ds_write2st64_b32(成对 256B 步长写)** | ❌ 多块 72/72 全坏;单块首 launch 坏后好(lds_st64_multi/lds_asm) |
| rocBLAS Tensile fp32(829 等) | ❌ 全尺寸垃圾 |
| **hsa_queue_create(新队列)** | ❌ 0x1001/0x1004 波动,内核静默(qtest) |
| hipModuleGetFunction(该 .co) | ❌ 段错误 |

**推断**: KFD/HSA 运行时资源路径在重启后退化(新队列创建失败/不稳 → 依赖自有队列的 rocBLAS 全灭;hip 初始化期队列仍可用 → naive/fp16 幸存)。LDS 成对指令故障独立存在。**建议:完整断电(拔电源 1 分钟)验证硬件可恢复性;不行则 fp32 原生路径在本卡上判死,应用层转置重组兜底不变。**

**Bjlk 修复中间产物**(本轮): patch_bjlk.py(829 机制+转置全局访问)、inject_927.py(dat 注入新 solution)、Cijk_Ailk_Bjlk_SB_MT32x8x32_SN_K1_patched.s + 编译好的 .hsaco —— GPU 恢复后可直接验证(927 已注入 fixed-lib dat,原 dat 备份 .bak-927)。

### 2026-08-12 16:00 最终结论(断电验证后)

**完整断电(拔电源 1 分钟)未恢复** → 判定:GPU 计算路径硬件级退化(重启 4 次 + 断电均无效)。

**幸存路径(全部实测)**:
- fp16/bf16 mm(hipBLASLt): ✅ 全尺寸正常 — ComfyUI 推理不受影响
- fp32 mm ≤512(torch 走 hipBLASLt): ✅ 正常
- fp32 mm ≥768: ❌ 上午已存在的 hipBLASLt 周期4 bug(非本次退化)
- rocBLAS/Tensile fp32(829/465/927): ❌ 全灭 — fixed-lib 工作暂废

**新增诊断发现**:
- LDS <2KB + b64 加载: 全坏;≥2KB + b64: 全好(lds_b64sizes) — 与 write2st64 故障同族,疑 LDS 低地址区双端口/向量化单元损坏
- hsa_queue_create: 链接 libamdhip64 后必失败(0x1004);纯 HSA 下正常 — hip 静态构造器与 KFD 队列分配交互损坏
- GPU agent 的 pool 列表缺 coarse VRAM 池(只剩 2×fine + LDS 64KB)

**后续建议**: 换卡(RX6600+ 全支持)或接受 fp16-only;若坚持 fp32,应用层兜底仅剩 hipBLASLt ≤512 或 CPU。

### 2026-08-12 17:00 重大修正(用户质疑"硬件退化"后复测)

**上一节"GPU 计算栈退化/硬件级"结论全部作废 —— 是测试方法错误的误判。**

**真实情况(全部用 torch 复测,正确环境 = LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib + ROCBLAS_TENSILE_LIBPATH=fixed-lib):**
- **GPU 硬件 100% 正常**(无需换卡/售后)
- **fixed-lib 成果完好**: a@b(829)64~5504 全对、a@b.t()(465)全对、nn.Linear/x@W.t 全对
- **Bjlk 从未坏过**: 802 在 torch 路径下 Linear 2.4e-6、x@W.t 1.5e-5~4.9e-4 —— 上午"43/43 系统性全坏"是 **C 直调(ctypes/gemm_multi)假象**(kernargs 差异),违反了 README 自己的教训"验证必须用 torch 本身"
- **x.t()@w 的错 = torch 传非连续 A**(transa=N+stride 转置)——系统版也一样错,与 kernel 无关;x.t().contiguous()@w 正常
- 系统版 rocBLAS(无 LD_LIBRARY_PATH)≥768 确实错(坏 kernel 抢占)—— 这正是 fixed-lib 修掉的;≤512 系统版其实正常

**下午误判的错误链(教训)**: ① torch 测试忘了 LD_LIBRARY_PATH(系统版 6.4.4-4 与定制版 dat 不兼容)→ 全错;② C 直调测试(RUNPATH 钉死定制版 + kernargs 差异)→ 更错;③ 断电前后没换环境复测。**正确的环境变量组合是 fixed-lib 生效的前提,缺一不可。**

**907 补丁(我的 Bjlk hybrid)**: 与 802 等效(都不需要修)—— 保留在 bjlk-diff/ 作产物,dat 已恢复原始(.bak-927)。

### 2026-08-12 18:00 结论修正总表(旧章节保留仅作历史)

以下旧结论基于 **C 直调(ctypes/gemm_multi/gemm_test803)或错误环境(系统版 rocBLAS 缺 LD_LIBRARY_PATH)**,已实测推翻:

| 旧结论 | 修正 |
|---|---|
| Bjlk(transB=T)43/43 系统性全坏, 生成器模板 bug | ❌ 作废 — 802 在 torch 路径 nn.Linear 2.4e-6、x@W.t 全尺寸正常; 重编 829 同款配置也无需修 |
| bmm fp32 全坏(64×64 也错, 转置重组无效) | ❌ 作废 — fixed-lib 下 8x128x64=1.1e-5、2x512x512=6.1e-5 正常 |
| 系统库 512³ 错 25% / 全面不可用 | ⚠️ 部分 — 系统版(6.4.4-4)≥768 方阵确坏(坏 kernel 抢占, fixed-lib 已修); ≤512 其实正常 |
| 重编 kernel 与官方 bit 级同错 | ❌ 作废 — 829/465 重编后 torch 路径全尺寸 bit 精确 |
| "GPU 计算栈退化/硬件级"(下午误判) | ❌ 作废 — 硬件 100% 正常, 见上节 |

**唯一可靠的验证方法: torch 本身 + 正确环境** (source rocblas-env.sh)。
**C 直调(rocblas_sgemm 直接调)结果一律不可信** — kernargs 差异导致与 torch 路径行为不同(829 直调错、torch 对)。
**正确环境 = LD_LIBRARY_PATH=/opt/rocm-6.4.3/lib + ROCBLAS_TENSILE_LIBPATH=fixed-lib + HSA_OVERRIDE_GFX_VERSION=8.0.3**, 三件套缺一不可。

## 2026-08-12 21:00-22:30 更新: ComfyUI SD1.5 端到端跑通(三层修复)

**目标**: 用 SD1.5 跑图验证显卡。发现 3 层独立错误, 全部定位并修复, 最终出图正常。

### 错误 1: fp16 分支缺失 (HIPBLAS_STATUS_NOT_SUPPORTED)

- 现象: KSampler 阶段报 `No Tensile solution found ... a_type f16_r, transA T, M1280 N2 K320`。
- 根因: 8-12 重建 fixed-lib 时**只保留了 Float/Double 分支, Half 匹配表被删空**。torch fp16
  matmul 走 rocBLAS gemm_ex 时无 solution 可匹配。README 之前"fp16 一直正常"是 hipBLASLt
  路径的结论, 而自编译 torch 2.8 **没编 hipBLASLt 后端** (`_get_hipblaslt_version` 不存在,
  libtorch_hip.so 有 hipblasLt* 符号但运行时不用), fp16 实际全走 rocBLAS Tensile。
- 修复: `inject_hh.py` — 把系统版 4.4.1 的 8 个 HH/HH_HPA fallback 分支 (168 solutions,
  index 927+) 注入 fixed-lib 的 TensileLibrary.dat, 复制 8 个 gfx803 hsaco (kernel symbol
  名 = solution name, rocBLAS 预加载 `*gfx803*co` 后按名匹配)。**已验证**: fp16 linear
  M=1280/1024/320/5504 全尺寸 err 0.015~0.03 (正常 fp16 累积误差)。

### 错误 2: fp16 小尺寸坏 kernel (乱码根源之一)

- 现象: 修复错误 1 后图仍是彩色噪声乱码 (无报错)。
- 定位: 逐 op 对比 A(系统版)/B(fixed-lib+HH) 数值:
  - **系统版 fp16 误差巨大**: M320×K320 err=6913、M1024 err=173 → 系统版 HH kernel 也坏!
  - **fixed-lib fp16 大尺寸正常** (0.015), 但 **小尺寸坏**: 64×64 err=31、128×128 err=**1e6**、
    s@v dim=32 err=5566。小尺寸命中 997/998 (GRVW2/VW2) 坏 kernel, 大尺寸命中 1000/1027
    (GRVW1/VW1) 好 kernel。
- 结论: 系统版 HH kernels 在 gfx803 上好坏混杂 (与 fp32 相同模式), 注入时把坏的也带进来了。
- 当前绕过: `--force-fp32` (fp32 在 fixed-lib 下全尺寸 1e-4 验证过), 慢但可靠。
- **待办**: 像 fp32 白名单 (829/465/802) 一样筛选 fp16 好 kernel 重建匹配表, 恢复 fp16 速度。

### 错误 3 (总根源): torch 原生 layer_norm kernel 在 gfx803 上数值全错

- 现象: fp32 模式仍乱码。hook 逐层对比 GPU/CPU: time_embed/proj_in/conv 全对, 但
  **LayerNorm (norm1/2/3) 输出 err 0.7~1.9** — 每个 transformer block 都算错。
- 定位: 直接对比 torch 原生 vs 手动实现:
  - `torch.nn.functional.layer_norm` (GPU): err=**2.56** ❌ (系统版环境也一样 → 与
    rocBLAS/fixed-lib 无关, 是 torch 编译进 HIP 的原生 kernel 在 gfx803 上坏)
  - 手动 `(x-mean)*rsqrt(var+eps)` 基于 torch mean/var 原语: err=**9e-7** ✅
  - GroupNorm (torch 原生) 正常 (4.8e-7), 只有 LayerNorm 坏。
- 修复: `~/ComfyUI/fix_layernorm_wrapper.py` — monkey-patch `F.layer_norm` (及
  `torch.layer_norm`) 为手动 mean/var 实现 (仅 CUDA 路径; CPU 走原实现)。启动入口改为
  wrapper。**已验证全形状 1e-6**。
- 坑: patch 不能只改 F.layer_norm 还要改 torch.layer_norm; CPU 分支必须调保存的
  `_orig_torch_layer_norm` (直接调 _orig F.layer_norm 会经 torch.layer_norm 递归爆栈)。

### 当前可用配置 (systemd: comfyui-gfx803.service)

```
ExecStart=.../fix_layernorm_wrapper.py --listen 127.0.0.1 --port 8188 --lowvram --use-split-cross-attention
```
- **纯 fp16 推理** (白名单后不再需要 --force-fp32): SD1.5 20 步 512×512 42 秒出图正常。
- `--use-split-cross-attention`: attention_split 内部有 .contiguous(), sub_quad 没有
  (非连续输入触发 transA=T+transB=T 双转置, 匹配表无此布局)
- 实测: SD1.5 20 步 512×512, 42 秒出图, 红苹果+木桌, 与提示词完全一致。

### 遗留问题 / 根治方向 (按性价比)

1. ~~fp16 坏 kernel 白名单筛选~~ **已完成 (2026-08-12 23:00)**: `whitelist_fp16.py`
   迭代删除坏 solution — 只删了 **Ailk_Bljk HPA=True 家族的 10 个** (1017-1020,
   1023-1027, 1030), 其他 7 个分支全部保留。删除后:
   - fp16 全 shape 扫描: 除 bmm_sv K=4096 的 0.1248 外全部 OK — 该值经核对是
     **fp16 大 K 累积的正常误差** (rel 3.9e-4, 与系统版逐位一致), 非坏 kernel。
   - **去掉了 --force-fp32, SD1.5 20 步 512×512 纯 fp16 42 秒出图正常。**
   - 备份: `TensileLibrary.dat.pre-whitelist`。工具: `whitelist_fp16.py` (迭代式,
     TENSILE_DB 抓选中 solution → 删除 → 重扫)。
   - 结论: 坏 kernel 规律 = 部分 GRVW2/VW2 变体在 Ailk_Bljk 布局坏, 但**不能只凭
     特征猜** — 必须 torch 实测 (README 老教训)。实际坏集比预想小得多。
2. **LayerNorm 根因调查**: torch 原生 layer_norm HIP kernel 在 gfx803 全错但 group_norm
   对 — 可能是自编译 torch 的 kernel 用到了 gfx803 不支持的特性 (ds_write2st64 同族?),
   重编 torch 指定 offload-arch=gfx803 或换 kernel 实现可根治, 无需 wrapper。
3. **hipBLASLt 后端**: 重编 torch 开 USE_HIPBLASLT 后 fp16 走 hipBLASLt (README 记录
   hipBLASLt fp16 全对), 可同时解决错误 2 和错误 3 之外的 fp16 路径。成本最高。
4. **换卡 RX6600+**: 最终方案, 全支持无坑。

### 诊断命令速查 (新增)

```bash
# LayerNorm 数值验证 (GPU vs CPU)
# 原生 (应为错): torch.nn.functional.layer_norm(x, (320,), w, b)
# 手动 (应为对): (x-x.mean(-1,keepdim=True))*torch.rsqrt(x.var(-1,unbiased=False,keepdim=True)+eps)

# fp16 各尺寸错误率扫描 (找坏 kernel)
~/ComfyUI/venv_gfx803/bin/python /tmp/small_mat.py   # 小尺寸 err 扫描

# UNet 前向 GPU vs CPU 逐层 diff (定位错层)
# hook_compare.py / hook2.py / hook3.py (见会话记录, 都在 /tmp/)
```
