# DeepSeek-V4 的百万上下文注意力设计：CSA 与 HCA

> 本文围绕 DeepSeek-V4 的百万 token 上下文注意力设计展开，重点分析长上下文推理中的 KV cache 与 attention 计算瓶颈，以及 CSA/HCA 在 Hybrid Attention 架构中的分工。

## 1. DeepSeek-V4 概览

DeepSeek-V4 是一个面向百万 token 上下文的 MoE 模型系列，官方发布形态可按两个维度理解：一是 Flash / Pro 两个规模，二是 Base / post-trained 两类训练阶段。所有版本都支持 1M token context，并共享同一套开源 inference 实现；规模差异主要体现在 `hidden_size`、层数、attention heads 数量、专家规模、`index_topk` 和 `compress_ratios` 层间配置上。

| 模型 | 总参数 | 激活参数 | Context length | 精度 | 定位 |
| --- | ---: | ---: | ---: | --- | --- |
| DeepSeek-V4-Flash-Base | 284B | 13B | 1M | FP8 Mixed | Flash 规模的 base model，用于基础能力评测和后续训练 |
| DeepSeek-V4-Flash | 284B | 13B | 1M | FP4 + FP8 Mixed | Flash 规模的对话 / 推理模型，强调低成本长上下文使用 |
| DeepSeek-V4-Pro-Base | 1.6T | 49B | 1M | FP8 Mixed | Pro 规模的 base model，用于基础能力评测和后续训练 |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M | FP4 + FP8 Mixed | Pro 规模的对话 / 推理模型，面向更高能力上限 |

其中 Base 表示基础模型版本，主要用于预训练后能力评测和进一步对齐训练；不带 Base 的 Flash / Pro 则是经过后训练的模型版本，支持 non-thinking、thinking 和 Think Max 等推理模式。

从发布时间看，DeepSeek-V4 属于 DeepSeek-V3 之后的一次大版本架构更新。Hugging Face 官方博客在 2026-04-24 发布 DeepSeek-V4 介绍；作为对照，Hugging Face Transformers 文档记录 DeepSeek-V3 发布于 2024-12-27。按公开发布日期计算，二者间隔约 483 天，即约 16 个月。

相比 DeepSeek-V3，DeepSeek-V4 本次更新的重点更集中在百万上下文推理效率、长轨迹 agent 任务和 KV cache 成本控制上。

从官方模型页面给出的评测看，V4-Pro 在知识、长上下文和 agentic 任务上明显强于 V3.2-Base；在与闭源前沿模型比较时，V4-Pro-Max 的优势集中在代码、长上下文和部分 agentic benchmark，整体表现接近前沿模型。

| 评测项 | DeepSeek-V3.2-Base | DeepSeek-V4-Flash-Base | DeepSeek-V4-Pro-Base | 观察 |
| --- | ---: | ---: | ---: | --- |
| MMLU-Pro, EM | 65.5 | 68.3 | 73.5 | Pro-Base 相比 V3.2-Base 有明显提升 |
| SimpleQA Verified, EM | 28.3 | 30.1 | 55.2 | Pro-Base 在事实问答上提升最显著 |
| FACTS Parametric, EM | 27.1 | 33.9 | 62.6 | 长期知识记忆与参数化事实能力增强 |
| HumanEval, Pass@1 | 62.8 | 69.5 | 76.8 | 代码生成能力提升 |
| LongBench-V2, EM | 40.2 | 44.7 | 51.5 | 长上下文 benchmark 提升 |

| 评测项 | Claude Opus 4.6 Max | GPT-5.4 xHigh | Gemini 3.1 Pro High | DS-V4-Pro Max | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| MMLU-Pro, EM | 89.1 | 87.5 | 91.0 | 87.5 | 接近 GPT-5.4，但低于 Gemini 3.1 Pro |
| GPQA Diamond, Pass@1 | 91.3 | 93.0 | 94.3 | 90.1 | 高难推理仍略低于主要闭源模型 |
| LiveCodeBench, Pass@1 | 88.8 | - | 91.7 | 93.5 | 代码 benchmark 上表现突出 |
| MRCR 1M, MMR | 92.9 | - | 76.3 | 83.5 | 1M 长上下文能力高于 Gemini 3.1 Pro，但低于 Opus 4.6 |
| SWE Verified, Resolved | 80.8 | - | 80.6 | 80.6 | 软件工程任务接近闭源前沿 |
| Terminal Bench 2.0, Acc | 65.4 | 75.1 | 68.5 | 67.9 | agentic terminal 任务处于第二梯队 |

DeepSeek-V4 依然主打性价比。DeepSeek 官方 API 文档显示，V4-Flash 和 V4-Pro 均按 1M tokens 计价，Pro 价格在 2026-05-31 15:59 UTC 后正式调整为原价的 1/4。按输出价格比较，DeepSeek-V4-Pro 为 $0.87 / 1M tokens，约为 GPT-5.4 的 5.8%、Claude Opus 4.6 的 3.5%、Gemini 3.1 Pro Preview 的 4.8%-7.3%。

| 模型 | 上下文长度 | 输入，cache miss | 输入，cache hit / cached | 输出 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek-V4-Flash | 1M | $0.14 / 1M tokens | $0.0028 / 1M tokens | $0.28 / 1M tokens | DeepSeek 官方 API |
| DeepSeek-V4-Pro | 1M | $0.435 / 1M tokens | $0.003625 / 1M tokens | $0.87 / 1M tokens | DeepSeek 官方 API，价格按标准费用的 1/4 计算 |
| GPT-5.4 | 270K 以下标准价 | $2.50 / 1M tokens | $0.25 / 1M tokens | $15.00 / 1M tokens | OpenAI 官方 API |
| Claude Opus 4.6 | 1M | $5.00 / 1M tokens | $0.50 / 1M tokens | $25.00 / 1M tokens | Anthropic 官方 API，标准模式 |
| Gemini 3.1 Pro Preview | <=200K | $2.00 / 1M tokens | $0.20 / 1M tokens | $12.00 / 1M tokens | Google Gemini API 标准价 |
| Gemini 3.1 Pro Preview | >200K | $4.00 / 1M tokens | $0.40 / 1M tokens | $18.00 / 1M tokens | Google Gemini API 标准价 |

上述评测与价格定位背后，是 DeepSeek-V4 对长上下文能力、训练稳定性和推理成本的一组系统设计：

![DeepSeek-V4 架构局部：CSA/HCA 位于 Transformer block 的 attention 路径](assets/csa_hca_article/network/v4_fig2_architecture_cropped.png)

这张结构图说明了 DeepSeek-V4 中几类关键模块的位置关系。虚线框表示重复堆叠的 Transformer block：右下方的 CSA/HCA 位于 attention 路径，负责 token 间信息交互；右上方的 DeepSeekMoE 位于前馈网络路径，用于提升参数容量和计算效率；左侧的 Residual Mixing 以及中间的 Pre-Block / Post-Block Mixing 对应 mHC 相关连接，用于改善层间信息传递。输入 token 经过 embedding 后进入多层 block，最终由 prediction head 输出下一 token 分布。

| 工作 | 简要说明 | 主要优化对象 |
| --- | --- | --- |
| **Hybrid Attention, CSA/HCA** | **在 attention 路径中混合 Compressed Sparse Attention 和 Heavily Compressed Attention，降低长上下文 attention 计算与 KV cache 成本。** | **Sequence / block 维度的历史 KV 访问与缓存** |
| DeepSeekMoE | 在前馈网络中使用 MoE 结构，通过稀疏激活提升模型容量与计算效率。 | FFN 参数容量与每 token 激活计算量 |
| Manifold-Constrained Hyper-Connections, mHC | 增强 residual / block mixing 连接，改善深层网络的信息流动。 | 层间表示传递与训练稳定性 |
| Muon Optimizer | 用于提升大规模训练过程中的收敛效率与稳定性。 | 优化器与训练动态 |
| 低精度与推理实现 | 结合 FP8、FP4、sparse attention kernel 和缓存管理降低服务成本。 | 推理显存、带宽、吞吐与延迟 |

本文重点分析第一项，即 **Hybrid Attention 中的 CSA 与 HCA**。HCA 指 Heavily Compressed Attention，属于注意力机制；mHC 指 Manifold-Constrained Hyper-Connections，属于连接结构增强机制。二者名称相近，但作用位置和优化对象不同。

长文档问答、代码仓库级理解、长链路 agent 工作流、多轮工具调用以及 test-time scaling，都要求模型在很长的历史上下文中保存、检索和整合相关信息。因此，在讨论具体优化机制之前，需要先明确 1M context 下 KV cache 与 attention 计算的主要瓶颈。

## 2. 长上下文的核心瓶颈：KV cache 与 attention 计算

长上下文推理需要先区分 prefill 和 decode 两个阶段。Prefill 阶段一次性处理输入 prompt，并为每层 attention 生成历史 token 的 K/V；decode 阶段每次只生成一个新 token，并在每一步复用已经缓存的历史 K/V。本文讨论的主要瓶颈出现在 decode 阶段：上下文越长，单步生成需要保留、读取和匹配的历史 K/V 越多。

![Transformer decode 阶段的 KV cache 读写路径：当前 token 生成 Q/K/V，历史 K/V 从 cache 复用并追加新 K/V](assets/csa_hca_article/network/brown_transformer_decode_kv_cache.png)

上图展示了单个新 token 在 Transformer attention 中的计算路径。当前 token 生成新的 $q_t/k_t/v_t$；$q_t$ 用来访问已经缓存的历史 key，历史 value 提供被加权读取的内容；当前生成的 $k_t/v_t$ 再追加回 KV cache，供后续 token 使用。由于每一层 attention 都维护自己的 KV cache，cache 的 sequence 维度会随着上下文长度增长。

以一层 decoder attention 为例，KV cache 的典型 shape 可写为：

$$
K,V \in \mathbb{R}^{B \times n \times h_{kv} \times d_h}
$$

因此，全模型 KV cache 存储量近似为：

$$
M_{\mathrm{KV}} = 2L B n h_{kv} d_h b
$$

其中 $B$ 是 batch size，$n$ 是历史序列长度，$h_{kv}$ 是 KV head 数，$d_h$ 是 head dimension，$L$ 是层数，$b$ 是每个元素的存储字节数。前面的 $2$ 分别对应 K 和 V。这个公式说明，KV cache 与 $n$ 线性相关；batch size、层数、KV head 数、head dimension 和存储精度决定每个 token 的缓存成本，$n$ 决定需要缓存的 token 数。长上下文场景的核心压力来自 sequence 维度长度的持续增长。

除了存储和读取 KV cache，decode 阶段每生成一个 token，还需要用当前 query 与历史 key 计算 attention score，并基于 score 从历史 value 中聚合信息：

$$
\begin{aligned}
s_t &= q_t K_{1:t}^{\mathsf T}, \\
o_t &= \operatorname{softmax}(s_t) V_{1:t}.
\end{aligned}
$$

$q_t$ 表示当前 token 的 query，$K_{1:t}$ 和 $V_{1:t}$ 表示到当前位置为止的历史 key/value。历史越长，$q_t$ 需要匹配的 key 越多，后续加权读取的 value 也越多。因此，decode 单步 attention FLOPs 和 KV cache 读取量都会随历史长度近似线性增长。对所有层做一个简化估算，单个 output token 的 attention FLOPs 量级可写为：

$$
F_{\mathrm{attn/token}} \approx 4L h_q d_h n
$$

其中 $h_q$ 表示 query head 数。这里的 $4$ 来自 $QK^{\mathsf T}$ 和 $AV$ 两次矩阵向量乘的 multiply-add 量级；该估算只覆盖 attention score 计算和 value 加权求和，softmax、RoPE、kernel 调度和内存访问开销需在 profiling 中单独统计。

下面以 DeepSeek-V4-Flash 为例，在以下假设下估算长上下文 decode 成本：batch size $B=1$，4-way tensor parallel；配置取 $L=43$、$h_q=64$、$d_h=512$、$h_{kv}=1$，对应 `num_hidden_layers=43`、`num_attention_heads=64`、`head_dim=512`、`num_key_value_heads=1`，KV cache 按 FP8 计算。

| Context length | Attention FLOPs / output token | KV cache 存储量，GB |
| ---: | ---: | ---: |
| 4K | 0.023 TFLOPs | 0.180 GB |
| 32K | 0.185 TFLOPs | 1.443 GB |
| 128K | 0.739 TFLOPs | 5.771 GB |
| 1M | 5.910 TFLOPs | 46.171 GB |

表中 KV cache 按十进制 GB 计算。数值用于量级比较，实际值会随 batch size、并行方式、KV 精度、attention kernel 和缓存布局变化。上下文从 4K 扩展到 1M 后，单步 attention FLOPs 从约 0.023 TFLOPs 增至约 5.910 TFLOPs，单请求 KV cache 从约 0.180GB 增至约 46.171GB，二者均呈约 256 倍增长。Flash 的 `num_key_value_heads=1` 和 FP8 存储已经显著降低了 KV cache 绝对值；在更大的 batch size、更多 KV heads 或更高 KV 精度下，缓存容量会继续线性上升。即便缓存容量低于权重规模，decode 每步仍要从显存读取越来越长的历史 K/V，因此带宽压力会随上下文长度同步增加。

因此，长上下文 attention 至少包含两个相关但不同的瓶颈：

| 瓶颈 | 主要问题 | 影响 |
| --- | --- | --- |
| KV cache 内存和带宽 | 历史 K/V 的存储量和读取量随序列长度增长 | 上下文越长，缓存容量和带宽压力越高 |
| Attention FLOPs | query 需要与大量历史 key 计算匹配分数 | 每生成一个 token，都需要处理更长的历史序列 |

多种 attention 优化方法均围绕同一目标展开：在尽量保留有效信息的前提下，降低 KV 存储、KV 读取和 attention 计算成本。从 tensor 维度看，压缩可发生在多个方向：减少 KV head 数量、降低每个 token 的 KV 表示宽度、减少每次 attention 访问的 token/block 数量，或沿 sequence/block 维度将多个 token 的 KV 合并为 compressed KV entry。下一节将按压缩维度比较几类常见 attention 路线。

## 3. Attention 压缩路线：从 MHA 到 DeepSeek MLA/DSA

在分析 CSA/HCA 之前，有必要先比较几类 attention 压缩路线。MHA 是标准多头注意力基线；MLA 和 DSA 则是 DeepSeek 团队围绕长上下文效率提出的两类代表性工作，分别作用于 hidden/latent 表示维度和 sequence/block 访问维度。这个对比可以帮助理解 DeepSeek-V4 为什么继续沿 sequence 维度压缩 KV。

![MHA、MQA/GQA、MLA、DSA 与 CSA/HCA 分别压缩不同 tensor 维度](assets/csa_hca_article/03_compression_routes.svg)

### MHA/MQA/GQA：KV head 维度的压缩

标准 MHA 中，每个 attention head 都拥有独立的 query、key 和 value 投影。单个 attention head 的基本计算路径是：由输入生成 Q/K/V，经由 $QK^{\mathsf T}$ 得到 attention score，再对 V 做加权求和。

![标准注意力 QKV 计算关系：QK^T 生成 attention score，再对 V 加权求和](assets/csa_hca_article/supplemental/standard_attention_qkv.webp)

在多头形式下，上述路径会在多个 heads 上并行执行。MHA、GQA 和 MQA 的主要差异体现在 KV head 的共享方式：MHA 为每个 query head 保留独立 K/V；MQA 让全部 query heads 共享一组 K/V；GQA 介于二者之间，让一组 query heads 共享一组 K/V。这个变化直接减少 KV cache 的 head 维度宽度。

![MHA、GQA 与 MQA 的 KV head 共享关系](assets/csa_hca_article/network/gqa_paper_mha_mqa_gqa.png)

设输入 hidden states 为：

```text
H: [n, d]
```

其中 `n` 是序列长度，`d` 是 hidden size。MHA 投影出多组 K/V：

```text
K, V: [n, h, d_head]
```

`h` 表示 head 数量。该结构的优势在于各个 head 可以学习不同的匹配模式；其代价是 KV cache 规模与 `n`、`h` 和 `d_head` 同时相关。在长上下文 decode 场景中，缓存容量和读取带宽都会随序列长度显著增加。

MQA/GQA 通过让多个 query head 共享部分或全部 KV head 来降低 KV cache 成本。以 MQA 为例，其 KV 表示可简化为：

```text
Q: [n, h_q, d_head]
K, V: [n, 1, d_head]
```

该方法主要压缩 KV head 维度，能够降低 cache 成本，但 sequence 维度仍保持为 `n`，query 仍需处理完整历史序列。

### DeepSeek MLA：压缩表示维度

MLA, Multi-head Latent Attention，首次出现在 DeepSeek-V2，随后被 DeepSeek-V3/V3.2 等模型沿用。它的核心思路可以理解为 LoRA 风格的低秩 KV 压缩：先用 down projection 把 hidden state 压到低维 latent，再用 up projection 生成 attention 需要的 K/V。KV cache 中保存低维 latent，避免为每个 token 存储完整展开后的多头 K/V。

![DeepSeek-V2 MLA/MoE 架构图：MLA 将 KV 相关表示压缩到 latent representation](assets/csa_hca_article/network/wikimedia_deepseek_mla.svg)

![MLA 低秩 KV 压缩：cache 中保存低维 c_KV，需要时恢复 K/V](assets/csa_hca_article/supplemental/mla_low_rank_compression.png)

给定 token 表示 $h_t$，KV 侧的低秩路径可简化写为：

$$
\begin{aligned}
c_t^{KV} &= W^{DKV} h_t, \\
k_t^C &= W^{UK} c_t^{KV}, \\
v_t^C &= W^{UV} c_t^{KV}.
\end{aligned}
$$

其中 $W^{DKV}$ 是降维矩阵，$W^{UK}$ 和 $W^{UV}$ 是升维矩阵，$c_t^{KV}$ 是需要缓存的 compressed latent KV。这个结构等价于将原本直接从 $h_t$ 生成 K/V 的大矩阵拆成低秩矩阵乘积：

$$
W_K \approx W^{UK} W^{DKV}, \qquad
W_V \approx W^{UV} W^{DKV}.
$$

这正是 LoRA 常见的 low-rank decomposition 思路：用一个较小 rank 的中间表示降低参数和中间状态成本。区别在于，MLA 把这种低秩结构放进 attention 的主路径，并把低维 $c_t^{KV}$ 作为 KV cache 的核心状态。

实际 DeepSeek MLA 还会将带 RoPE 的 key 分量与低秩 KV 分支解耦。原因是 RoPE 引入位置信息，难以直接吸收到上述低秩矩阵乘积中；推理时缓存主要由 $c_t^{KV}$ 和少量位置相关 key 状态组成，整体仍显著小于完整 multi-head K/V cache。

从 tensor shape 看，MLA 的 KV cache 可以概括为：

$$
H \in \mathbb{R}^{n \times d}
\quad\longrightarrow\quad
C^{KV} \in \mathbb{R}^{n \times d_c},
$$

其中 $d_c$ 小于原本展开后的 KV 表示维度。因此，MLA 主要压缩 hidden/latent 表示维度，降低每个 token 需要保存的 KV 表示规模。

从 sequence 维度看，MLA 仍然为每个 token 保留一份 latent 表示：

$$
\text{sequence length}: n \rightarrow n,\qquad
\text{representation dim}: d_{kv} \rightarrow d_c.
$$

因此，MLA 主要降低每个 token 的 KV 表示宽度，sequence 维度长度仍保持为完整序列长度。

### DeepSeek DSA：减少访问的 token/block 数量

DSA, DeepSeek Sparse Attention，首次作为 DeepSeek-V3.2-Exp 的核心架构改动出现，后续 DeepSeek-V3.2 继续沿用。它是 DeepSeek 团队面向长上下文稀疏访问提出的 attention 设计，压缩方向转向 sequence/block 访问维度，目标是减少 query 实际访问的 token/block 数量。实现这类稀疏访问的关键组件是 Lightning Indexer：先用轻量索引器为历史 token/block 打分，再把 top-k 结果交给主 attention 计算。

![NVIDIA TensorRT-LLM DSA 架构图：用 Lightning Indexer 和 top-k 选择需要访问的 token/block](assets/csa_hca_article/network/nvidia_dsa_architecture.png)

Lightning Indexer 可以理解为一个轻量的 attention-like 检索器。它也有 query/key 匹配过程，但使用的是更低维的 indexer query/key，只负责估计“哪些 token/block 值得主 attention 访问”，不会直接生成最终 attention output。对于当前 query token $t$ 和候选 token/block $s$，其简化打分过程如下：

$$
I_{t,s} = \operatorname{score}(q_t^I, k_s^I),
\qquad
\mathcal{S}_t = \operatorname{TopK}_s(I_{t,s}, k).
$$

其中 $q_t^I$ 是当前 token 派生出的低维 indexer query，$k_s^I$ 是候选 token/block 的低维 indexer key，二者的维度小于主 attention 使用的 K/V 表示维度。$\mathcal{S}_t$ 是被选中的 top-k 集合。主 attention 随后只访问 $\mathcal{S}_t$ 中的 K/V：

$$
o_t = \operatorname{Attention}(q_t, K_{\mathcal{S}_t}, V_{\mathcal{S}_t}).
$$

因此，被压缩的是 sequence/block 访问维度：候选集合从完整序列 $[1,2,\ldots,n]$ 缩小到 top-k 相关 token/block。该方法能够降低 attention FLOPs，尤其适用于 decode 阶段。

DSA 的有效性依赖 Lightning Indexer 的选择质量。如果索引器遗漏关键信息，后续 attention 计算会失去该信息。因此，sparse attention 的效果同时取决于稀疏化比例、索引器选择精度、选择粒度以及局部上下文补偿机制。DeepSeek-V4 的 CSA 延续这一路径，但索引对象从原始 token/block 转为 compressed KV entries。

| 方法 | 主要压缩维度 | 主要缓解的问题 | 仍然留下的问题 |
| --- | --- | --- | --- |
| MHA | 基本不压缩 | 表达直接，结构清楚 | KV cache 大，长上下文 attention 计算重 |
| MQA/GQA | KV head 维度 | 降低 KV cache 中 head 维成本 | sequence 维度仍保持为完整历史长度 |
| MLA | hidden/latent 表示维度 | 降低每个 token 的 KV 表示成本 | sequence 长度仍保持为完整上下文 |
| DSA | sequence/block 访问维度 | 减少每次 attention 访问的 token/block 数量 | 依赖 Lightning Indexer 选择质量，局部细节需要补偿 |

基于上面的路线，可以看到 attention 压缩至少涉及三类维度：KV head 维度、hidden/latent 表示维度，以及 sequence/block 访问维度。DeepSeek-V4 的 CSA/HCA 继续沿 sequence/block 维度压缩 KV entries，将多个 token 的 KV 合并为一个 compressed KV entry，并在此基础上组合稀疏选择、滑动窗口和层间配置。至于 V4 是否仍保留 MLA 的低秩投影思想，需要等 CSA/HCA 机制介绍完成后，再结合代码统一说明。

## 4. CSA：KV 压缩与稀疏选择

CSA, Compressed Sparse Attention，可以看作 DeepSeek-V4 在 DSA 稀疏访问思路上的进一步推进：先把长序列 KV 沿 sequence/block 维度压短，再在压短后的候选集合中选择相关 compressed entries。这样，主 attention 访问的 KV 由两部分构成：最近一段 uncompressed sliding window，以及被 indexer 选中的 long-range compressed entries；shared KV MQA 与 grouped output projection 则继续降低 head 维度和输出投影成本。

本节按一次 decode 的数据流展开：

1. token-level compressor 将每 `m` 个 token 的 KV 聚合为 compressed KV entry。
2. Lightning Indexer 为 compressed entries 打分，并选择 top-k 相关 entries。
3. sliding window 保留最近 token 的原始粒度 KV。
4. core attention 将 top-k compressed KV 与 sliding window KV 拼接后执行 attention，再经过 grouped output projection 回到 hidden size。

![DeepSeek-V4 官方 CSA 数据流：compression、lightning indexer、top-k、sliding window 与 shared KV MQA](assets/csa_hca_article/network/v4_fig3_csa.png)

![vLLM c4a attention 动画：展示压缩 KV、sliding window 与稀疏访问的动态过程](assets/csa_hca_article/network/vllm_c4a_animation.gif)

从 tensor 角度看，CSA 的输入可表示为：

$$
H \in \mathbb{R}^{n \times d}.
$$

`n` 是当前序列长度，`d` 是 hidden size。CSA 会通过 token-level compressor 生成 compressed KV：

$$
C^{comp} \in \mathbb{R}^{\lceil n/m \rceil \times c}.
$$

`m` 表示压缩率，`c` 表示压缩后每个 KV entry 的维度。该过程发生在 sequence/block 维度：原始 `n` 个位置被压缩为约 `n/m` 个 compressed entries。

### Learned gated pooling：区别于平均池化

CSA 的压缩采用 learned gated pooling。模型首先由 hidden states 生成待压缩的 KV entry 和压缩权重：

![CSA token-level compressor：overlap compression 缓解 block 边界信息割裂](assets/csa_hca_article/supplemental/csa_token_compressor.png)

$$
C = H W_{KV},\qquad Z = H W_Z.
$$

其中 `C` 表示候选 KV 表示，`Z` 表示用于决定压缩权重的 score，`W_KV` 和 `W_Z` 均为可学习参数。该机制允许模型为 block 内不同 token 和不同维度分配不同权重，避免对 block 内 token 做无差别平均。

对于长度为 `m` 的 block，压缩过程可简化表示为：

$$
S_{i,j}=\operatorname{softmax}_j(Z_{i,j}+B_j),
\qquad
C_i^{comp}=\sum_j S_{i,j} C_{i,j}.
$$

`B` 表示可学习的位置 bias，`S_j` 表示第 `j` 个 token 在压缩中的权重。该公式表示，同一 block 内的 token 会以不同权重写入 compressed KV entry，权重由模型学习得到。

CSA 还采用 overlapped compression。该机制保持压缩后的 entry 数量约为 `n/m`，同时让相邻 block 的信息在压缩时发生交叠，缓解硬切块造成的边界割裂。

开源 inference 代码中，`compress_ratio == 4` 时 `overlap=True`。此时 `wkv` 和 `wgate` 输出 `2 * head_dim`，比无 overlap 时多出一份压缩特征，可以理解为两路分支：

```text
normal branch:  当前 token 写入本 block 的 compressed entry
overlap branch: 当前 token 写入下一个 block 的 overlapped compressed entry
```

以 `m=4` 为例，若原始序列被切成：

```text
B0 = [t0, t1, t2, t3]
B1 = [t4, t5, t6, t7]
B2 = [t8, t9, t10, t11]
```

普通 non-overlap compression 会让 `C0` 只来自 `B0`，`C1` 只来自 `B1`，`C2` 只来自 `B2`。overlapped compression 下，除第一个 block 外，每个 compressed entry 会同时接收前一个 block 的 overlap branch 和当前 block 的 normal branch：

```text
C0 <- normal(B0)
C1 <- overlap(B0) + normal(B1)
C2 <- overlap(B1) + normal(B2)
```

也就是说，`B0` 中的 token 会以 normal branch 参与 `C0`，也会以 overlap branch 参与 `C1`；`B1` 同理既参与 `C1`，也参与 `C2`。在代码实现中，这对应 `overlap_transform`：当前 block 的后一半维度写入自己的压缩窗口，前一个 block 的前一半维度写入下一个压缩窗口；score 也按同样方式移动，并在合并时做 softmax 加权池化。

这个设计有两个直接效果。第一，相邻 block 边界附近的信息不会完全落入两个互不相通的 compressed entries。第二，压缩后的 sequence length 仍然约为 `n/m`，不会因为 overlap 变成两倍；增加的是每次压缩时参与聚合的候选特征和相应投影成本。HCA 的 `compress_ratio=128` 不启用 overlap，因为其目标是更激进地生成粗粒度全局表示，继续扩大每个 entry 的覆盖范围会进一步降低信息粒度。

### Lightning indexer：在压缩块里选 top-k

获得约 $\lceil n/m \rceil$ 个 compressed entries 后，若每个 query 对全部 compressed entries 执行 dense attention，计算量已经低于原始 $n$ 个 token 的 dense attention，但在百万上下文场景下仍可能较高。因此，CSA 进一步引入 indexer。

![CSA Lightning Indexer：查询压缩块、计算索引得分并选择 top-k](assets/csa_hca_article/supplemental/csa_lightning_indexer.png)

Indexer 为 compressed KV blocks 生成独立的 compressed indexer keys：

$$
K_I^{comp} \in \mathbb{R}^{\lceil n/m \rceil \times c_I}.
$$

`c_I` 表示 indexer head dimension。对于 query token `t`，indexer 从 query hidden state 生成低秩 indexer query：

$$
h_t \rightarrow c_t^Q \rightarrow q_t^I.
$$

随后对每个 compressed block `s` 计算相关性分数：

$$
I_{t,s}=\operatorname{score}(q_t^I, K_{I,s}^{comp}).
$$

`I_{t,s}` 表示 query token `t` 与 compressed block `s` 的相关性。技术报告和代码中还包含多 indexer head、ReLU、head weight 等细节。其功能是在进入 core attention 之前，以较低成本筛选相关 compressed blocks。

随后 CSA 根据 index score 选择 top-k compressed entries：

$$
\mathcal{S}_t=\operatorname{TopK}_s(I_{t,s}, k),
\qquad
C_t^{sparse}=\{C_s^{comp}\mid s\in\mathcal{S}_t\}.
$$

该步骤减少 attention 访问数量：query 仅访问 top-k compressed entries，避免扫描全部 `n/m` 个 compressed entries。

### Sliding window：局部细节与因果边界补偿

CSA 同时保留一段 uncompressed sliding window KV，主要原因包括两点。

第一，compressed block 存在因果边界。query 只能访问其之前已经完成压缩的 blocks，同一 block 中未来 token 的细粒度信息保持不可见。第二，语言建模对近邻 token 高度敏感，部分局部依赖更适合保留原始粒度。

因此，实际 attention 的 KV 来源由两部分组成：

$$
KV_t^{attended}
=\operatorname{concat}\left(KV_t^{window}, C_t^{sparse}\right).
$$

$KV_t^{window}$ 保留最近 `n_win` 个原始粒度 token，$C_t^{sparse}$ 提供更远历史的压缩检索结果。DeepSeek-V4-Flash 的配置里 `sliding_window` 是 128，开源 inference 代码中 `window_size` 也对应这个概念。

以 DeepSeek-V4-Flash 接近 1M context 的 decode 为例，可以把上述步骤具体化。Flash 的 CSA 层使用 `compress_ratio=4`、`index_topk=512`、`sliding_window=128`。当历史长度为 1,048,576 tokens 时：

| CSA 步骤 | 数量变化 | 含义 |
| --- | ---: | --- |
| Token-level compression | 1,048,576 tokens -> 262,144 compressed entries | 名义上每 4 个 token 生成 1 个 compressed KV entry |
| Lightning indexer | 262,144 candidates -> top-512 entries | 用低成本索引器从压缩块中选出相关块 |
| Sliding window | 额外保留最近 128 个原始 token | 保留局部细节，并处理压缩块的因果边界 |
| Core attention | 512 compressed entries + 128 raw entries = 640 entries | 当前 query 在压缩后的较小工作集上执行主 attention |

这里的 640 表示主 attention 实际访问的 KV entries 数量。512 个 compressed entries 各自携带若干 token 的聚合信息；overlapped compression 还会让相邻 block 的信息进入压缩过程，从而缓解硬切分造成的边界损失。

### Shared KV MQA 与 grouped output projection

CSA 的 core attention 使用 shared key-value MQA：compressed KV entry 同时作为 key 和 value，并被多个 query heads 共享。该设计进一步降低 KV head 维度上的成本。

![CSA shared KV MQA 与 grouped output projection：共享 KV 并分组降低输出投影成本](assets/csa_hca_article/supplemental/csa_shared_kv_mqa_grouped_output.png)

由于 query heads 的输出拼接后维度较大，直接执行完整输出投影会引入较高计算成本。DeepSeek-V4 因此采用 grouped output projection：先将 attention heads 分组，每组投影到较小的中间表示，再合并投回 hidden size。Flash 配置中的 `o_groups=8`、`o_lora_rank=1024` 对应该机制，相关代码位于 `Attention.forward` 的 `wo_a` 和 `wo_b` 路径。

CSA 同时降低 KV cache 长度和长程 attention 访问数量：

$$
\text{KV cache length}: n \rightarrow \lceil n/m\rceil,
\qquad
\text{attention visited entries}: \lceil n/m\rceil \rightarrow k+n_{win}.
$$

因此，CSA 同时作用于 KV cache 规模和长程 attention 计算量。

## 5. HCA：更高压缩率下的全局注意力

HCA, Heavily Compressed Attention，与 CSA 共享若干基础组件：二者都执行 KV compression，都采用 shared KV MQA 和 grouped output projection，并且都保留 sliding window KV 以补偿局部细节。

![DeepSeek-V4 官方 HCA 数据流：重压缩后在 compressed entries 上做 dense attention](assets/csa_hca_article/network/v4_fig4_hca.png)

关键差异在于：HCA 使用更高压缩率，但不执行 sparse top-k selection。

从 tensor shape 看，HCA 将原始长度 `n` 压缩为：

```text
C_comp: [n / m', c]
```

其中 `m'` 远大于 CSA 的 `m`。技术报告指出 HCA 使用更大的压缩率 `m' >> m`，并采用 non-overlapped compression。在 Flash 与 Pro 配置中，`4` 对应较轻压缩的 CSA 层，`128` 对应重压缩的 HCA 层；二者都以 CSA/HCA 交替作为主要层间组织方式，但边界层配置不同。

继续使用 1,048,576 tokens 的 decode 示例，HCA 层在 `compress_ratio=128` 时的数据流更直接：

| HCA 步骤 | 数量变化 | 含义 |
| --- | ---: | --- |
| Heavy compression | 1,048,576 tokens -> 8,192 compressed entries | 每个 compressed entry 覆盖约 128 个 token |
| Indexer / top-k | 无 | 省去 compressed entries 的 top-k 选择 |
| Sliding window | 额外保留最近 128 个原始 token | 补偿局部细节和因果边界 |
| Core attention | 8,192 compressed entries + 128 raw entries = 8,320 entries | 当前 query 对压缩后的全局历史执行 dense attention |

这个例子说明了 HCA 与 CSA 的差别：HCA 访问的 compressed entries 多于上面 CSA 示例中的 top-512，但它覆盖的是完整长历史，并省去了 indexer 和 top-k 选择过程。代价是每个 compressed entry 的粒度更粗，单个 entry 需要概括更长的 token span。

HCA 的压缩同样使用 learned weights：

```text
C = H · W_KV
Z = H · W_Z
S = softmax(Z + B)
C_comp_i = sum_j S_j * C_j
```

变量含义与 CSA 相同：`C` 表示候选 KV，`Z` 表示压缩权重的 score，`B` 表示可学习位置 bias。不同之处在于，HCA 的每个 compressed entry 覆盖更长 token span，因此压缩后的 sequence length 更短。

压缩完成后，HCA 省去 indexer top-k 选择，直接在 heavily compressed KV entries 上执行 dense attention：

```text
o_t = Attention(q_t, C_comp, C_comp)
```

其中 `C_comp` 同时作为 key 和 value。CSA 更侧重在较细粒度 compressed blocks 中选择相关块；HCA 则将长历史压缩为更粗粒度的全局表示，使 query 能够以较低成本访问全局上下文。

因此，HCA 更适合理解为 compressed dense attention。HCA 的长程分支在 `n/m'` 个 heavily compressed entries 上执行 dense attention。例如，当 1M token 使用 `m'=128` 时，长程 entries 数量约为 8192，显著少于原始 token 数。

需要再次区分 HCA 与 mHC：HCA 是 attention 中的重压缩全局分支；mHC 是 residual connection 相关机制。后续讨论聚焦 HCA 与 CSA 在 Hybrid Attention 中的分工。

## 6. 从 MLA 到 CSA/HCA：DeepSeek-V4 保留了什么、改写了什么

理解 CSA 与 HCA 的数据流之后，可以回到一个容易混淆的问题：DeepSeek-V4 是否还在使用 MLA。结合 DeepSeek-V4 inference 代码看，答案需要拆开看：Q/O 路径仍保留 MLA 的低秩投影思想；长程 KV cache 主路径则从 DeepSeek-V2/V3 的 per-token latent KV，转向 CSA/HCA 的 sequence/block compressed KV entries。

V2/V3 MLA 的 KV 路径可以概括为：

$$
\begin{aligned}
c_t^{KV} &= W^{DKV} h_t, \\
k_t &= W^{UK} c_t^{KV}, \\
v_t &= W^{UV} c_t^{KV}.
\end{aligned}
$$

缓存对象是每个 token 的低维 $c_t^{KV}$，需要时再通过上投影恢复 K/V。因此，MLA 主要压缩每个 token 的 KV 表示宽度，sequence length 仍然保持为 $n$。

DeepSeek-V4 的代码显示，Q/O 路径仍然保留 low-rank projection。`Attention.__init__` 中的 query 路径是：

```python
self.wq_a = Linear(self.dim, self.q_lora_rank)
self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
```

对应到 Flash 配置，query 从 `hidden_size=4096` 先降到 `q_lora_rank=1024`，再升到 `num_attention_heads * head_dim = 64 * 512`。Pro 配置中则是 `hidden_size=7168 -> q_lora_rank=1536 -> 128 * 512`。这是 LoRA/MLA 风格的低秩投影路径，但 Q 只参与当前步计算，不进入 KV cache。

输出投影也保留了类似的低秩分解。代码中先用 grouped `wo_a` 将每组 attention heads 的输出投到 `o_lora_rank`，再用 `wo_b` 投回 hidden size：

```python
self.wo_a = ColumnParallelLinear(
    self.n_heads * self.head_dim // self.n_groups,
    self.n_groups * args.o_lora_rank
)
self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
```

Flash 中 `o_groups=8`、`o_lora_rank=1024`；Pro 中 `o_groups=16`、`o_lora_rank=1024`。这说明 V4 仍在 Q/O 投影上延续低秩化思想，以降低大规模多头 attention 输出投影的计算和参数成本。

KV 路径的变化更关键。V4 没有沿用 V2/V3 MLA 的 `c_t^{KV} -> up projection -> K/V` 作为长程 KV cache 主路径。用于 sliding-window 的原始 KV 由：

```python
self.wkv = Linear(self.dim, self.head_dim)
```

生成，并经 `kv_norm`、RoPE 和量化后写入窗口缓存。Flash 中这相当于 `4096 -> 512`，Pro 中是 `7168 -> 512`。由于 `num_key_value_heads=1`，这个 512 维 KV 表示会被多个 query heads 共享，属于 shared KV / MQA 风格的 KV 表示压缩。

当层类型为 CSA 或 HCA 时，V4 进一步调用 `Compressor` 对长程 KV 做 sequence/block 压缩。`Compressor` 中的核心参数是：

```python
self.wkv = Linear(self.dim, coff * self.head_dim)
self.wgate = Linear(self.dim, coff * self.head_dim)
```

其中 `coff=1+overlap`。CSA 的 `compress_ratio=4` 启用 overlap，Flash 中 compressor 的 `wkv` 等价于 `4096 -> 1024 = 2 * 512`；HCA 的 `compress_ratio=128` 不启用 overlap，等价于 `4096 -> 512`。这些投影确实在 hidden_size 维度上做降维，但它们不同于 MLA 的 $W^{DKV}$：后续路径没有 $W^{UK}/W^{UV}$ 将 latent 展开成每个 token 的 K/V。投影结果会在 block 内通过 learned gated pooling 聚合：

$$
C_i^{comp} = \sum_j S_{i,j} C_{i,j},
$$

然后作为 compressed KV entry 直接写入 cache，并在 core attention 中同时作为 key 和 value 使用。

CSA 的 Indexer 还体现了另一层低维检索思想。`Attention.forward` 会先计算：

```python
qr = q = self.q_norm(self.wq_a(x))
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
```

其中 `qr` 是经过 `wq_a` 和 `q_norm` 后的低秩 query 中间态。CSA 的 `Indexer` 复用这个 `qr`，再通过自己的 `wq_b` 投到 indexer heads：

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
```

这里的 `head_dim` 是 indexer 的 `index_head_dim`，Flash 与 Pro 都是 128，小于主 attention 的 `head_dim=512`。Indexer 还拥有自己的 `Compressor(args, compress_ratio, self.head_dim, True)`，用于生成 128 维 compressed indexer keys。随后它用：

```python
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

为 compressed blocks 打分并选择 top-k。也就是说，CSA 的稀疏选择先用更低维的 indexer query/key 做一次低成本检索，避免直接用主 attention 的 512 维 K/V 做全量匹配，再把结果交给主 attention。HCA 没有这条 Indexer 路径；ratio 128 层会用 `get_compress_topk_idxs` 返回全部 compressed entries，等价于在 heavily compressed entries 上做 dense attention。

因此，可以把 DeepSeek-V4 对 MLA 思想的继承关系总结为四层：

| 位置 | DeepSeek-V4 中的体现 | 与 V2/V3 MLA 的关系 |
| --- | --- | --- |
| Q projection | `wq_a -> q_norm -> wq_b`，先降到 `q_lora_rank` 再升到多头 Q | 保留 low-rank projection 思想 |
| O projection | grouped `wo_a -> wo_b`，按组做低秩输出投影 | 保留 low-rank projection 思想，并适配多头输出 |
| KV cache | `wkv` 先把 hidden 投到 shared KV，再由 `Compressor` 沿 sequence/block 聚合为 compressed KV entries | 继承“压缩 KV 状态”的目标，取消 per-token latent KV up-projection 路径，转向 sequence/block 压缩 |
| CSA indexer | 复用低秩 query 中间态 `qr`，再投到 128 维 indexer heads，用低维 compressed keys 选 top-k | 延续“用低维表示降低检索成本”的思想，但服务于稀疏选择 |

V4 将 MLA 的“低秩投影”和“减少 KV 状态成本”拆开使用：Q/O 侧继续低秩化，KV 侧从 per-token latent compression 升级为 CSA/HCA 的 block-level KV compression。这样做的直接收益是，V4 不只降低每个 token 的 KV 表示宽度，还进一步减少长上下文中需要长期保存和访问的 KV entry 数量。

## 7. 混合注意力架构：CSA 与 HCA 如何分工

DeepSeek-V4 采用 CSA 与 HCA 的混合配置，以利用二者在长程信息访问上的互补关系。

![DeepSeek-V4-Flash 的 CSA/HCA 层间交替示意](assets/csa_hca_article/06_layer_schedule.svg)

CSA 采用较轻压缩和稀疏选择，保留更细的 long-range block 粒度，适合进行相对精细的长程检索。当 query 只需要历史中的少数相关片段时，CSA 的 indexer 可筛选这些 compressed blocks。

HCA 采用重压缩和 compressed dense attention，无需 top-k 选择少数块即可以较低成本提供粗粒度全局覆盖。

仅使用 CSA 会引入两类成本：其一，indexer 和 top-k sparse attention 本身具有额外计算开销；其二，层层依赖稀疏选择会持续要求模型判断相关历史块。仅使用 HCA 也存在局限，因为过高压缩率会降低长程信息粒度，从而影响细粒度检索能力。

二者混合后形成如下分工：

| 分支 | 典型压缩率 | 长程访问方式 | 主要作用 |
| --- | --- | --- | --- |
| CSA | 较小，例如配置中的 4 | top-k sparse over compressed entries | 较细粒度长程检索 |
| HCA | 较大，例如配置中的 128 | dense over heavily compressed entries | 低成本全局覆盖 |
| Sliding window | 不压缩 | dense over recent tokens | 局部细节和因果边界补偿 |

从复杂度角度看，CSA 的长程 attention 访问数量受 top-k 上限约束，主要成本与被选中的 compressed entries 数量相关；HCA 的成本则来自对 `n/m'` 个 heavily compressed entries 执行 dense attention。对 full-sequence 视角而言，CSA 可近似理解为以 `top-k` 约束长程工作集，HCA 可近似理解为在长度 `n/m'` 的压缩序列上执行 dense attention。二者结合后，一个分支提供较精细的稀疏检索，另一个分支提供粗粒度全局覆盖。

Flash 与 Pro 的 `compress_ratios` 进一步说明了这种设计会随模型规格调整 layer schedule：

| 版本 | `compress_ratios` 数量 | 不压缩层 | CSA 层，ratio=4 | HCA 层，ratio=128 | 边界模式 | `index_topk` |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Flash | 44 | 3 | 21 | 20 | `[0, 0, 4, 128, ... , 4, 0]` | 512 |
| Pro | 62 | 1 | 30 | 31 | `[128, 128, 4, 128, ... , 4, 0]` | 1024 |

Flash 的开头两项和最后一项为 `0`；Pro 仅最后一项为 `0`，并且前两项已经使用 HCA。因此，“不压缩层位于开头和结尾”只适用于 Flash 配置。更稳定的结论是：`4/128` 的交替体现了层间交替处理细粒度相关信息和粗粒度全局信息的设计思路；具体边界层如何配置，则随模型版本和规模变化。

因此，Hybrid Attention 是围绕 sequence compression、sparse selection、sliding window 和 layer schedule 展开的系统设计。

## 8. 从论文公式到推理代码：CSA/HCA 的实现对应

技术报告给出机制定义，开源 inference 代码进一步展示了这些机制在推理路径中的具体组织方式。本节按数据流对应 `Compressor`、`Indexer`、`Attention.forward` 和 `sparse_attn` 等关键模块。

![CSA/HCA 论文概念与 inference 代码模块的对应关系](assets/csa_hca_article/07_code_mapping.svg)

![vLLM c4a decode path：operator graph、kernel fusion 与多 stream 执行路径](assets/csa_hca_article/network/vllm_decode_path.svg)

### ModelArgs 与 config.json

Flash 与 Pro 配置显示，两者共享同一套 `inference/model.py` 和 `inference/kernel.py`，CSA/HCA 的实现路径一致；模型规格差异主要由 `config.json` 驱动：

| 配置项 | Flash | Pro | 对应概念 |
| --- | ---: | ---: | --- |
| `num_hidden_layers` | 43 | 61 | Transformer 层数 |
| `hidden_size` | 4096 | 7168 | 主干表示维度 |
| `num_attention_heads` | 64 | 128 | query heads 数量 |
| `num_key_value_heads` | 1 | 1 | shared KV MQA 配置 |
| `max_position_embeddings` | 1048576 | 1048576 | 1M context |
| `sliding_window` | 128 | 128 | sliding window KV 长度 |
| `compress_ratios` | 44 项，0/4/128 | 62 项，0/4/128 | 不压缩层、CSA 层、HCA 层 |
| `head_dim` | 512 | 512 | compressed KV / attention head 维度 |
| `qk_rope_head_dim` | 64 | 64 | 部分 RoPE 维度 |
| `index_topk` | 512 | 1024 | CSA indexer 选择的 top-k |
| `index_head_dim` | 128 | 128 | indexer head 维度 |
| `n_routed_experts` | 256 | 384 | MoE routed experts 数量 |
| `num_experts_per_tok` | 6 | 6 | 每 token 激活专家数 |
| `o_groups` | 8 | 16 | grouped output projection 分组数 |

推理代码中的 `ModelArgs` 字段名与配置文件略有差异。例如，`window_size` 对应配置中的 `sliding_window`，`rope_head_dim` 对应 `qk_rope_head_dim`。

### Compressor：token-level KV 到 block-level KV 的压缩

`inference/model.py` 中的 `Compressor` 对应技术报告中的 token-level compressor。其核心参数包括：

```python
self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim))
self.wkv = Linear(self.dim, coff * self.head_dim)
self.wgate = Linear(self.dim, coff * self.head_dim)
self.overlap = compress_ratio == 4
```

`wkv` 生成待压缩 KV，`wgate` 生成压缩权重，`ape` 表示 learnable positional bias。`compress_ratio == 4` 时启用 overlap，对应 CSA 的 overlapped compression；`compress_ratio == 128` 时不启用 overlap，对应 HCA 的重压缩流程。

在 forward 路径中，压缩核心可概括为：

```python
kv = self.wkv(x)
score = self.wgate(x)
score = score + self.ape
kv = (kv * score.softmax(dim=2)).sum(dim=2)
```

该实现对应 learned gated pooling：每个 token 写入 compressed entry 的权重由 `score` 决定，而非平均分配。

decode 阶段需要增量维护尚未形成完整 block 的 token。代码中的 `kv_state` 和 `score_state` 用于缓存未压缩尾部状态，因为单个新 token 到达时不一定立即形成新的 compressed KV entry。

### Indexer：只在 CSA 层出现的稀疏选择器

`Indexer` 只在 `compress_ratio == 4` 时被创建：

```python
if self.compress_ratio == 4:
    self.indexer = Indexer(args, self.compress_ratio)
else:
    self.indexer = None
```

在 Flash 与 Pro 的 inference 代码中，ratio 4 对应 CSA，需要 learned top-k selection；ratio 128 对应 HCA，不创建 indexer。

`Indexer` 内部包含独立的 `Compressor`，用于生成 compressed indexer keys。随后，它从 query 侧生成 indexer query：

```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
```

随后与 compressed indexer keys 计算 index score：

```python
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
```

该过程对应技术报告中的 lightning indexer：先为 compressed blocks 计算相关性分数，再返回 `topk_idxs`，供后续 sparse attention kernel 执行 KV gather。

代码还对 indexer 的 Q/K 路径执行 FP4 simulation：

```python
fp4_act_quant(q, fp4_block_size, True)
```

该细节对应技术报告中 indexer 使用低精度计算以降低长上下文成本的设计。

### Attention.forward：window KV 与 compressed KV 的汇合

`Attention.forward` 是 CSA/HCA 数据流的汇合位置。该函数首先生成 query 和当前 sliding window KV：

```python
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
kv = self.wkv(x)
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
```

`get_window_topk_idxs` 生成最近窗口内可访问的位置。若当前层包含 compression 分支，则继续生成 compressed indexes：

```python
if self.indexer is not None:
    compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
else:
    compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
```

该分支体现 CSA 与 HCA 的实现差异：

- CSA 层：`self.indexer is not None`，使用 learned indexer 生成 top-k compressed indexes。
- HCA 层：不存在 indexer，`get_compress_topk_idxs` 返回所有已完成的 compressed entries，等价于在 heavily compressed entries 上执行 dense attention。

最终，attention kernel 接收统一格式的 `topk_idxs`：

```python
o = sparse_attn(q, kv_or_cache, self.attn_sink, topk_idxs, self.softmax_scale)
```

因此，sliding window、CSA top-k 和 HCA 的全部 compressed entries 都被组织为基于 index 的 KV gather 形式，并进入同一个 sparse attention kernel。HCA 在概念上是 dense over compressed entries；在实现上，只要 index 列表包含全部 compressed entries，就可复用 indexed gather 路径实现等价计算。

### sparse_attn：基于 index 的 KV gather 与 online softmax

`inference/kernel.py` 的 `sparse_attn` 对应执行 attention 的 TileLang kernel。其输入包括：

```python
q: [b, m, h, d]
kv: [b, n, d]
topk_idxs: [b, m, topk]
```

kernel 根据 `topk_idxs` 将 KV gather 到 tile 中，并对每个 query 执行点积、softmax 和加权求和。实现使用 running max / running sum 形式的 online softmax，避免一次性 materialize 大量 attention score。同时，kernel 引入 `attn_sink`，对应技术报告中的 attention sink 设计。

低精度策略也体现在该路径中：非 RoPE 维度可执行 FP8 simulation，RoPE 的最后 64 维保留较高精度，以维持位置信息稳定。这与技术报告中 RoPE 维度使用 BF16、其余维度使用 FP8 的描述一致。

### KV Cache 管理与分层存储

Hybrid Attention 也改变了推理侧 KV cache 的组织方式。传统 PagedAttention 假设各层 KV cache 具有较一致的 block 管理策略；而 DeepSeek-V4 中不同层可能使用 CSA、HCA 或纯 sliding window，不同分支的 cache 生命周期、压缩率和访问模式并不相同。

![DeepSeek-V4 推理侧 KV Cache 管理：压缩缓存与状态缓存分离](assets/csa_hca_article/09_kv_cache_hierarchy.svg)

一种更合适的理解方式是将推理状态拆成两类：

| 缓存类型 | 保存内容 | 作用 |
| --- | --- | --- |
| Classic KV Cache | CSA/HCA 生成的 compressed KV entries | 保存可复用的长程压缩历史 |
| State Cache | sliding window KV 与尚未形成完整压缩块的尾部 token 状态 | 支持局部窗口和增量压缩 |

在 shared-prefix 复用场景中，compressed KV entries 的体积较小，可以作为更适合跨请求复用和分层存储的对象。SWA sliding window KV 体积更大、状态更短期，因此可以采用不同恢复策略：全量缓存最近窗口、周期性保存 checkpoint 并重算尾部，或完全不缓存并在需要时重算。该设计将长程压缩历史和局部状态分开管理，有助于降低百万上下文推理对 GPU 显存容量的依赖。

## 9. 效果评估：效率收益与长上下文表现

CSA/HCA 的主要作用在于降低百万上下文推理中的 attention 计算和 KV cache 成本。

![DeepSeek-V4 官方效率图：1M context 下 FLOPs 与 KV cache 降幅](assets/csa_hca_article/network/v4_fig1_efficiency.png)

![vLLM KV cache 对比图：DeepSeek-V3.2 与 DeepSeek-V4 的 per-layer KV state](assets/csa_hca_article/network/vllm_kv_cache_comparison.svg)

DeepSeek-V4 技术报告给出了 1M context 下相对 DeepSeek-V3.2 的效率对比：

| 模型 | 1M context single-token inference FLOPs | 1M context KV cache |
| --- | ---: | ---: |
| DeepSeek-V4-Pro | 约 27% | 约 10% |
| DeepSeek-V4-Flash | 约 10% | 约 7% |

报告中的 FLOPs 按 equivalent FP8 FLOPs 统计，比较对象是长上下文场景下的 single-token inference。上述收益来自一组组合设计，包括 hybrid CSA/HCA、低精度计算与存储、较小 top-k、混合 KV 存储格式等，不应归因于 CSA 或 HCA 的单一模块贡献。

长上下文能力方面，技术报告和 README 列出了 MRCR 1M、CorpusQA 1M、LongBench-V2 等结果。这些结果的意义在于说明模型不仅支持 1M context，还需要在该长度下保持可用能力，并将推理成本控制在更接近实际部署需求的范围内。

由此可归纳出以下设计启示：

- 仅压缩表示维度不足以解决长上下文成本问题，sequence 维度也需要被压缩或稀疏化。
- 稀疏选择需要与局部窗口和全局粗粒度覆盖配合，才能兼顾细粒度依赖和全局信息。
- 长上下文 attention 是系统设计问题，需要同时考虑压缩率、选择器、KV cache 管理、低精度格式和层间 schedule。

DeepSeek-V4 的 Hybrid Attention 可概括为三类机制的协同：CSA 负责较细粒度长程选择，HCA 负责低成本全局覆盖，sliding window 负责局部细节补偿。三者共同将百万上下文从“支持长输入长度”推进到“降低长上下文实际推理成本”。

## 参考资料

- DeepSeek-V4 Technical Report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- DeepSeek-V4-Pro Hugging Face model card and benchmark tables: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4 figures on Hugging Face Blog: https://huggingface.co/blog/deepseekv4
- DeepSeek-V3 Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/model_doc/deepseek_v3
- DeepSeek-V2 Technical Report: https://arxiv.org/abs/2405.04434
- DeepSeek-V3.2-Exp Release Note: https://api-docs.deepseek.com/news/news250929
- DeepSeek API Models & Pricing: https://api-docs.deepseek.com/quick_start/pricing
- OpenAI API Pricing: https://openai.com/zh-Hans-CN/api/pricing/
- Anthropic Claude API Pricing: https://platform.claude.com/docs/en/about-claude/pricing
- Google Gemini API Pricing: https://ai.google.dev/gemini-api/docs/pricing
- vLLM DeepSeek V4 engineering blog: https://vllm.ai/blog/2026-04-24-deepseek-v4
- vLLM FP8 KV Cache and Attention Quantization: https://vllm.ai/blog/2026-04-22-fp8-kvcache
- NVIDIA, Optimizing Inference for Long Context and Large Batch Sizes with NVFP4 KV Cache: https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- Brown CSCI1390 Lecture 7, Transformer FLOPs and KV caching: https://cs.brown.edu/courses/csci1390/notes/Lecture7_TransformerFlops.pdf
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints: https://arxiv.org/abs/2305.13245
- Pratham Patel, DeepSeek-V4 Hybrid Attention from Scratch: https://prathamp.com/blog/deepseek-v4-csa-and-hca/
- 老许漫谈 AIInfra，《DeepSeek V4 万字拆解：逼近闭源天花板，价格却只要 1/6》
- 补充图：标准注意力机制、MLA 低秩压缩、CSA token compressor、CSA lightning indexer、CSA shared KV MQA 与 grouped output projection
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- DeepSeek-V2 MLA/MoE diagram, DeepSeek, MIT License: https://commons.wikimedia.org/wiki/File:DeepSeek_MoE_and_MLA_(DeepSeek-V2).svg
- NVIDIA TensorRT-LLM DSA 架构说明调研: https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog17_Sparse_Attention_in_TensorRT-LLM.html
- DeepSeek-V4-Flash / Pro 开源配置与推理实现：`config.json`, `inference/model.py`, `inference/kernel.py`
