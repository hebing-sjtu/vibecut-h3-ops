# vibecut-h3-ops

把一个 MiniMax-H3 **Ref2VA** 请求包批量跑完，逐镜取回 MP4，并记录每镜实际用了什么参数。后端可以是自建 SGLang 服务，也可以是 FastVideo。

| 脚本 | 作用 |
|---|---|
| `run_pack.py` | **SGLang 路线**：预检 → 逐镜 POST `/v1/videos` → 轮询 → 下载 MP4 → 记 `runs.jsonl` |
| `run_pack_fastvideo.py` | **FastVideo 路线**：同样的包、同样的续跑与记账，但在进程内直接驱动 `VideoGenerator` |
| `probe_outputs.py` | 用 ffprobe 实测回传文件，产出 `probe.csv`：请求参数与实际编码结果并排 |
| `retarget_uris.py` | 把请求里的 `conditions[].uri` 改指到包实际所在的目录 |

```sh
git clone https://github.com/hebing-sjtu/vibecut-h3-ops.git
```

`run_pack.py`、`probe_outputs.py`、`retarget_uris.py` 只用 Python 3 标准库（`probe_outputs.py` 额外需要 `ffprobe`），clone 下来就能跑。`run_pack_fastvideo.py` 要在装了 FastVideo 的环境里跑，但它的 `--check` 预检不 import FastVideo，所以在任何机器上都能先验包。

## 请求包的形状

脚本不假设包里有什么内容，只依赖这个布局：

```
<pack>/
  requests/ref2va/01-xxx.json   # 每镜一个官方格式的请求
  requests/t2va/01-xxx.json     # 可选
  ...参考图等素材
```

每个请求是 MiniMax-H3 官方的 `/v1/videos` 请求体：

```json
{
  "task": "ref2va",
  "prompt": "subject_definitions:\n<Subject 1> is ...",
  "conditions": [
    { "type": "image", "uri": "file:///data/pack/refs/portrait.png", "role": "reference" }
  ],
  "target": { "short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 6 },
  "seed": 11
}
```

输出文件名取请求文件名：`requests/ref2va/01-studio.json` → `01-studio.mp4`。

## 目录约定

代码、素材、产物、权重建议分四处，互不写入对方。下面的命令都用这套变量：

```sh
export OPS=/workspace/vibecut-h3-ops              # 本仓库，可随时删了重拉
export PACK=/data/.../your-pack                   # 收到的素材包
export OUT=/data/.../your-pack-out                # 生成的 MP4、runs.jsonl、probe.csv
export HF_HOME=/data/.../models                   # 模型权重缓存
mkdir -p "$OUT" "$HF_HOME"
```

两个 `run_pack*` 脚本的 `--out` 默认都是 `<pack>/out/<task>`，也就是写进素材包里面。**建议每次显式传 `--out "$OUT/ref2va"`**，把产物放在包外的兄弟目录：重新解压素材包不会覆盖已经跑出来的镜头，校验包的哈希时也不会把自己的产物算进去。

## 让请求里的路径对上

**两条路线都要先做这一步。** `conditions[].uri` 由**跑模型的那个进程**解析，不是你敲 curl 的那台机器。包的默认 uri 通常指向作者约定的挂载点，解到别处就全是错的。

如果包自带重生成脚本（例如 `source/build_prompt_pack.py --mount-root "$PACK"`），优先用它，它会把提示词和请求一起重生成。没有的话用这里的：

```sh
python3 "$OPS/retarget_uris.py" --pack "$PACK"            # 先看要改什么
python3 "$OPS/retarget_uris.py" --pack "$PACK" --apply
```

它不需要你提供旧路径。每条 uri 从左往右逐段剥，保留能在新根下找到真实文件的最长后缀，所以只会改写到磁盘上确实存在的路径；`http(s)` 的远端参考原样保留。改完重跑是幂等的。

## 选哪条路线

两条路线都能做 Ref2VA，决定性的差别是**你的机器是 CUDA 12 还是 CUDA 13**。

| | SGLang | FastVideo |
|---|---|---|
| H3 支持起始版本 | v0.5.17（v0.5.16 还没有） | fork 的 `main` |
| torch / CUDA | ≥0.5.17 都要 `cuda-python>=13.0`，即 **CUDA 13**，驱动 ≥580 | `torch==2.12.0`，官方同时提供 **cu126 与 cu130** 两个 backend |
| 多参考的接口 | HTTP `/v1/videos`，`conditions[]` 有序 | 只有 Python API 的 `InputConfig(references=[...])` |
| 多机/多端点分发 | 起多份服务，`--base-url` 给多个 | 单进程占一组卡 |

**驱动低于 580 就只能走 FastVideo。** CUDA 13.x 要求驱动 ≥580，而容器里的驱动来自宿主、在容器内改不了；所有支持 H3 的 sglang 版本都没有 CUDA 12 分支，硬装会撞 `cuda-python>=13.0` 与 `torch==2.x+cu126` 的死结。

反过来，FastVideo 的 HTTP 层（`POST /v1/videos/generations`）只有单个 `input_reference`，表达不了有序多参考，所以它的 Ref2VA 只能从 Python API 进 —— 这就是 `run_pack_fastvideo.py` 不走 HTTP 的原因。

## 用法（FastVideo 路线）

### 1. 装环境

装在**新建的 venv** 里，不要动系统或镜像自带的环境：

```sh
git clone https://github.com/hebing-sjtu/FastVideo.git /workspace/FastVideo
uv venv /data/.../venvs/h3 --python 3.12 && . /data/.../venvs/h3/bin/activate
cd /workspace/FastVideo && uv pip install -e . --torch-backend cu126   # CUDA 13 的机器换 cu130
```

`--torch-backend` 只能由 `uv pip` 的命令行或 `UV_TORCH_BACKEND` 指定，写在 `pyproject.toml` 里没用。两个 backend 提供的都是 torch 2.12.0，所以换 backend 不动 CUDA 本身。

### 2. 跑

```sh
export HF_HOME=/data/.../models    # 否则几十 GB 权重落到 ~/.cache

python3 "$OPS/run_pack_fastvideo.py" --pack "$PACK" --check          # 先验包，不碰 GPU
python3 "$OPS/run_pack_fastvideo.py" --pack "$PACK" --out "$OUT/ref2va" \
  --num-gpus 8 --only 01,02                                          # 先看两镜的人物
python3 "$OPS/run_pack_fastvideo.py" --pack "$PACK" --out "$OUT/ref2va" \
  --num-gpus 8                                                       # 全量，已生成的跳过
```

`--num-gpus` 同时也是 sp 度，必须整除 H3 的 56 个注意力头，所以只能取 1/2/4/7/8/14/28；给 6 会在模型内部才炸，脚本因此在启动前就拦掉。模型只加载一次，九镜复用同一个 generator。

预检、续跑（已有 MP4 自动跳过）、单镜失败隔离、`runs.jsonl` 记账都和 SGLang 路线一致，`probe_outputs.py` 可以直接读它写的账。

**时长的处理和 SGLang 不同。** FastVideo 吃的是帧数不是秒数，所以脚本用 H3 自己的 `align_num_frames` 把 `duration_seconds` **向上**对齐到 17n+5：请求 6 秒 → 158 帧 = 6.583 秒，请求 8 秒 → 192 帧 = 8.000 秒。比请求长一点，不会短。请求值和对齐后的帧数都记在 `runs.jsonl` 里。

## 用法（SGLang 路线）

### 1. 起 Ref2VA 服务

```sh
uv pip install "sglang[diffusion]" --prerelease=allow

sglang serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant ref2va \
  --num-gpus 8 \
  --ulysses-degree 8 \
  --encoder-parallel auto \
  --performance-mode speed \
  --warmup-resolutions 1344x768 \
  --host 0.0.0.0 --port 30011
```

80 GB 的卡（H100/H800）装不下纯 Ulysses 的整条 pipeline，官方配方是 4 卡 `--tp-size 2 --ulysses-degree 2`；八张 80 GB 卡就用 `CUDA_VISIBLE_DEVICES` 起两份服务各占四张，端口 30011 / 30012。

**起服务的那个 shell 必须先 `export HF_HOME=...`**，否则几十 GB 权重会落到 `~/.cache/huggingface`，砸在系统盘上。

别把 `--model-path` 指向手动下载的子目录：checkpoint 目录映射由 SGLang 自己管，`--model-variant ref2va` 会去挑 `transformer_ref/`。首次启动自己走 Hub 解析，不需要预下载。

想先把权重拉好、把下载的失败模式和起服务的失败模式分开，就：

```sh
export HF_TOKEN=hf_...   # 或 hf auth login；匿名请求有限流，几十 GB 会很难受

# 先看会下哪些文件、一共多大，不真下
hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" --include "Ref2VA/*" --dry-run

hf download MiniMaxAI/MiniMax-H3 \
  --include "model_index.json" --include "Ref2VA/*"
```

**`--include` 必须每个模式写一次。** 新版 `hf` CLI 是 click 风格，一次 `--include` 只吃一个值，所以 `--include "a" "b"` 会把 `b` 当成位置参数里的文件名，然后报 `Ignoring --include since filenames have been explicitly set`，接着去下一个名字字面是 `b` 的文件并 404。等价的写法是用位置参数的子目录语法：`hf download MiniMaxAI/MiniMax-H3 model_index.json Ref2VA/`。

别加 `--local-dir` —— 那样不进缓存，SGLang 找不到，照样会重下一遍。

### 2. 预检

```sh
python3 "$OPS/run_pack.py" --pack "$PACK" --check
```

逐条确认 `task` 正确、prompt 非空、`t2va` 的 `conditions` 是空数组、每个本地参考文件真读得到，并列出每镜的参考图数量、seed 和 target。**必须是 `0 problem(s)` 再往下走。**

参考图读不到不会让服务报错，Ref2VA 会退化成没有身份约束的生成。所以预检默认拒绝开跑，`--skip-preflight` 只在服务端看到的是另一个挂载点（例如容器内路径）时才用。

### 3. 跑

```sh
# 先挑两镜确认人物
python3 "$OPS/run_pack.py" --pack "$PACK" --out "$OUT/ref2va" \
  --only 01,02 --base-url http://localhost:30011

# 全量；已生成的自动跳过，中断了直接重跑这条
python3 "$OPS/run_pack.py" --pack "$PACK" --out "$OUT/ref2va" \
  --base-url http://localhost:30011 --base-url http://localhost:30012

# 单镜重做，换种子
python3 "$OPS/run_pack.py" --pack "$PACK" --out "$OUT/ref2va" \
  --only 05 --force --seed 12345
```

`--base-url` 可以重复给。**每个端点同时只有一镜在飞**，这是刻意对齐服务端默认的 `batching_max_size: 1`；要并发就多起服务、`CUDA_VISIBLE_DEVICES` 占不相交的卡。

不带 `--seed` 就完全按包里的 seed 走，`target` 里的任何字段都不改写。

单镜失败只记账并继续，退出码非零，重跑时只补没成的那几镜。

## 记录实际参数（两条路线通用）

```sh
python3 "$OPS/probe_outputs.py" --dir "$OUT/ref2va"
```

```
file                  WxH    fps      dur  frames   audio      MB
01-studio.mp4    1344x768     24    5.208     124     aac     8.4
```

请求时长和实际编码时长本来就会不一样：H3 按 24 fps 的 17n+5 帧对齐，请求 6 秒回来不会正好 6.000 秒。要交付元数据时照 `probe.csv` 的实测值填，别抄请求值。`runs.jsonl` 里有每镜的 video id、seed、target 和墙上时间。

## 会咬人的地方

| 症状 | 原因 |
|---|---|
| 报找不到文件，但你本地 `ls` 得到 | `uri` 由服务进程解析。容器部署时官方 Docker 把宿主媒体目录只读挂在 `/data/minimax-h3`，包要放进挂载点内再重定向 uri |
| 人物像但服装/场景错位 | Ref2VA 的 `conditions` 顺序是语义化的，必须和 prompt 里的一基材料编号逐一对应。别重排参考图顺序 |
| 请求被拒 | `target.duration_seconds` 只接受 4–15 秒 |
| 画面比例不对 | Ref2VA 下 `aspect_ratio: "auto"` 落到模型的 16:9 兜底，不继承参考图几何。要别的比例就显式写 |
| 两份服务一起变慢或 OOM | 两份服务必须 `CUDA_VISIBLE_DEVICES` 占不相交的卡 |
| 轮询一直不结束 | `--shot-timeout` 默认 5400 秒。状态 GET 的瞬时失败会重试 10 次才放弃，避免代理 502 打掉一个正在跑的任务 |
| 装 sglang 撞 `cuda-python>=13.0` 解不开 | 支持 H3 的 sglang 全都只有 CUDA 13 分支。走 FastVideo 的 cu126 |
| FastVideo 里报头数除不尽 | `--num-gpus` 同时是 sp 度，必须整除 56 |
| FastVideo 只认一张参考图 | 它的 HTTP 层只有单个 `input_reference`。有序多参考只能走 Python API |
| 回来的时长比请求长半秒 | FastVideo 路线把秒数**向上**对齐到 17n+5 帧。请求 6 秒得到 6.583 秒 |

## 耗时参考

都是官方公布的数字，不同硬件差异很大，只能当量级：8×B300 单请求 Ref2VA 768p / 5 秒 / 50 步约 29 秒，模型加载另算约 114 秒；4×H200 同规格 T2VA 端到端约 74–84 秒，而 Ref2VA 大致是 T2VA 的 1.5 倍。

## 来源

请求与部署格式依据[官方 MiniMax-H3 仓库](https://github.com/MiniMax-AI/MiniMax-H3)与 [SGLang 的 MiniMax-H3 cookbook](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3)；FastVideo 路线的调用形状、帧数对齐与参考上限取自 [FastVideo](https://github.com/hebing-sjtu/FastVideo) 的 `examples/inference/basic/basic_minimax_h3_ref2va.py` 与 `fastvideo/pipelines/basic/minimax_h3/`（`align_num_frames`、`resolve_canvas_size`、`validate_references`）。核对日期 2026-09-13。本仓库不含模型权重，也不含任何素材包。
