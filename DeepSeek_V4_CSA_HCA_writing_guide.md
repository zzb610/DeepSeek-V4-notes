# DeepSeek-V4 的百万上下文注意力设计：CSA 与 HCA

> 写作目标：这是一篇面向技术分享的中文文章/报告，不是一篇论文复述。整体语言要平实、清楚、图文并茂，符合中文表达习惯，避免明显的 AI 味道。技术细节要讲透，但每个公式和 tensor shape 都要服务于直觉解释。

写作原则：

- 保持故事线的先后顺序。前文只提出当前问题和必要变量，后文才展开的概念不要提前解释；例如在“长上下文瓶颈”中只讨论 KV cache 和 attention 成本，不提前引入 MLA、DSA、CSA/HCA 的具体机制。
- 每一节只承担一个主要功能：先讲问题，再讲已有压缩路线，最后讲 DeepSeek-V4 的具体设计。跨节概念可以用一句话做过渡，但不要抢先展开。

## 写作主线

DeepSeek-V4 最值得作为切入点的地方，不只是模型规模变大，而是它把上下文长度推进到 1M token，同时大幅降低长上下文推理中的 attention FLOPs 和 KV cache 成本。本文以这个问题为主线：为什么长上下文会卡在 attention 上，已有的 MHA/MLA/DSA 分别压缩了什么，DeepSeek-V4 又为什么要把 CSA 和 HCA 混合起来使用。

需要特别注意两个边界：

- HCA 是 Heavily Compressed Attention，属于注意力机制的一部分。
- mHC 是 Manifold-Constrained Hyper-Connections，属于残差连接增强机制。介绍 DeepSeek-V4 基本情况时可以提到，但本文不要展开成主线，也不要和 HCA 混淆。

参考资料：

- 技术报告：`/Users/bowenyuchi/Projects/DeepSeek-V4/DeepSeek-V4-Flash/DeepSeek_V4.pdf`
- 如果本地 PDF 只是 Git LFS pointer，则参考官方报告链接：`https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf`
- 推理代码：`/Users/bowenyuchi/Projects/DeepSeek-V4/DeepSeek-V4-Flash/inference`
- 模型配置：`/Users/bowenyuchi/Projects/DeepSeek-V4/DeepSeek-V4-Flash/config.json`

## 1. 认识 DeepSeek-V4

本节目标：用相对轻松但信息密度足够的方式，让听众先知道 DeepSeek-V4 是什么、为什么值得关注，以及 CSA/HCA 在其中处于什么位置。

内容要点：

- DeepSeek-V4 是面向百万 token 上下文的 MoE 模型系列，包括 Pro 和 Flash。
- 给出模型规模：DeepSeek-V4-Pro 为 1.6T 总参数、49B 激活参数；DeepSeek-V4-Flash 为 284B 总参数、13B 激活参数。
- 给出上下文长度：两者都支持 1M token context。
- 简要介绍三个关键升级：Hybrid Attention、mHC、Muon Optimizer。
- 将焦点收束到本文主角：Hybrid Attention 中的 CSA 与 HCA。
- 可以用一张“DeepSeek-V4 架构总览图”开场，图中只标出 Embedding、CSA/HCA、MoE、mHC、Prediction Head，不要一开始就堆公式。

AI prompt：

```text
请写“认识 DeepSeek-V4”这一节，面向技术分享开场，语言自然、平实，不要像新闻稿。

必须包含：
1. DeepSeek-V4 系列包含 Pro 和 Flash 两个主要版本。
2. Pro/Flash 的总参数、激活参数和 1M token context。
3. DeepSeek-V4 的关键升级：Hybrid Attention、mHC、Muon Optimizer。
4. 明确说明本文只聚焦 Hybrid Attention 里的 CSA 和 HCA，mHC 只是背景信息。
5. 用一段话解释为什么百万上下文值得关注：它和长文档、长链路推理、agent 工作流、test-time scaling 的关系。

写作要求：
- 不要一上来进入公式。
- 不要把 HCA 和 mHC 混淆。
- 不要使用“革命性”“遥遥领先”等宣传口吻。
- 结尾自然过渡到“长上下文的核心瓶颈：KV cache 与 attention 计算”。
```

## 2. 长上下文的核心瓶颈：KV cache 与 attention 计算

本节目标：把问题讲清楚。听众需要先理解：为什么上下文从 128K 扩到 1M 后，attention 会成为推理效率的主要瓶颈。

内容要点：

- 从 autoregressive decoding 讲起：每生成一个 token，都要读历史 KV。
- 用 MHA 的 KV cache shape 解释内存随序列长度线性增长：`[batch, seq_len, num_heads, head_dim]`。
- 用 attention score 的计算解释 decode 阶段单 token attention 计算量随历史长度增长。
- 区分两个瓶颈：
  - KV cache 内存/带宽瓶颈：历史 KV 存得多、读得多。
  - attention 计算瓶颈：query 要和大量历史 key 做匹配。
- 引出“压缩”不是单一方向：可以压缩 head 维度、KV 表示维度、sequence 维度，也可以减少每次 attention 访问的 token/block 数量。

AI prompt：

```text
请写“长上下文的核心瓶颈：KV cache 与 attention 计算”这一节。

要求：
1. 从自回归生成的 decode 阶段讲起，说明每生成一个 token 都要访问历史 KV。
2. 用 MHA 的 tensor shape 解释 KV cache 为什么随 seq_len 增长。
3. 用简单公式或伪公式解释 attention 计算为什么随历史 token 数增长。
4. 明确区分 KV cache 瓶颈和 attention FLOPs 瓶颈。
5. 最后引出：后续技术路线本质上是在不同 tensor 维度上做压缩或稀疏化。

风格要求：
- 公式可以出现，但每个公式后必须用中文解释直觉。
- 举一个 1M context 的量级例子，但不要做未经核实的硬件性能推断。
- 不要写成教科书，要像给工程师做报告。
```

## 3. Attention 压缩路线：从 MHA 到 DeepSeek MLA/DSA

本节目标：介绍 CSA/HCA 之前的技术铺垫，但不要把它们写成“旧路线”。这里要强调一脉相承：不同方法在压缩 attention 的不同维度。

内容要点：

- MHA/MQA/GQA：用经典对比图说明 KV head 共享方式的差异；MHA 每个 attention head 都有自己的 K/V，MQA/GQA 共享部分或全部 KV head，从而降低 KV cache 的 head 维度成本。
- MLA：首次出现在 DeepSeek-V2，DeepSeek-V3/V3.2 继续沿用；把 KV 表示压到 latent space，需要讲清楚 LoRA 风格的低秩分解，即 down projection 得到低维 $c^{KV}$，再用 up projection 恢复 K/V；同时补一句 RoPE key 分量会解耦保存，重点说明它是在表示维度上压缩 KV。
- DSA：首次作为 DeepSeek-V3.2-Exp 的核心架构改动出现，后续 DeepSeek-V3.2 继续沿用；通过 Lightning Indexer 给候选 token/block 打分并选择 top-k，重点讲清楚 Lightning Indexer 是一个 attention-like 的轻量检索器，使用低维 indexer query/key 做便宜的匹配计算，并在 sequence/block 访问维度上减少主 attention 的计算。
- 总结对比：
  - MLA 主要减少 KV cache 表示成本。
  - DSA 主要减少 attention 访问数量。
  - CSA/HCA 进一步把 KV cache 沿序列维度压缩成 compressed KV entries。
- 为 CSA/HCA 做铺垫：DeepSeek-V4 将压缩 KV、稀疏选择、滑动窗口和混合层配置组合起来；V4 如何继承和改写 MLA，放到 CSA/HCA 机制讲完之后再结合代码说明。

AI prompt：

```text
请写“Attention 压缩路线：从 MHA 到 DeepSeek MLA/DSA”这一节。

必须讲清楚：
1. MHA/MQA/GQA 的基本差异：MHA 每个 query head 对应独立 K/V；MQA 所有 query heads 共享一组 K/V；GQA 在 query heads 分组内共享 K/V。
2. 说明这一组方法主要压缩 KV head 维度，能够降低 KV cache，但不减少 sequence/block 维度长度。
3. MLA 的出现版本与核心思想：首次出现在 DeepSeek-V2；用 LoRA 风格的低秩矩阵分解压缩 KV，先 down projection 得到 latent representation，再通过 up projection 恢复或使用 K/V；重点说明它主要压缩表示维度。
4. DSA 的出现版本与核心思想：首次作为 DeepSeek-V3.2-Exp 的核心架构改动出现；用 attention-like 的 Lightning Indexer 生成低维 indexer query/key 分数，保留 top-k 相关 token/block；重点说明它主要减少 sequence/block 访问数量。
5. 用一个小表格总结 MHA、MLA、DSA 分别压缩哪个维度、解决哪个瓶颈、留下什么问题。
6. 结尾引出 CSA/HCA：DeepSeek-V4 开始直接对序列维度上的 KV entries 做压缩；V4 如何继承 MLA 放到 CSA/HCA 机制之后再讲。

写作要求：
- 不要把 MHA/MLA/DSA 叫作“旧路线”或“过时方案”。
- 不要只讲概念，要用 tensor shape 帮助理解。
- 解释“维度”时要具体到 head 维、hidden/latent 维、sequence/block 维。
- 不要在本节直接展开 V4 代码级 MLA 继承关系，因为其中会用到 CSA/HCA、Compressor、Indexer 等后文概念。
```

## 4. CSA：先压缩 KV，再做稀疏选择

本节目标：详细解释 Compressed Sparse Attention 的数据流。核心是“每 m 个 token 的 KV 压成一个 compressed KV entry，然后用 learned indexer 选择 top-k compressed entries，再加上 sliding window 保留局部细节”。

内容要点：

- CSA 的总体流程：
  1. 输入 hidden states。
  2. token-level compressor 生成 compressed KV entries。
  3. lightning indexer 给 compressed blocks 打分。
  4. top-k selector 选出相关 compressed KV entries。
  5. 与 sliding window 的原始 KV entries 拼接。
  6. 做 shared key-value MQA。
- 不要在 CSA 主体中重新长篇回溯 MLA 历史或低秩投影；这些内容属于第 6 节。本节只在必要处承接“CSA/HCA 压缩的是 sequence/block 维度的 KV entries”。
- 解释压缩：
  - 每 `m` 个 token 压成 1 个 KV entry。
  - 技术报告中 CSA 还使用 overlapped compression：当前 block 和前一个 block 的信息共同参与。
  - 解释 overlap 的具体机制：`compress_ratio=4` 时 `wkv/wgate` 输出 `2 * head_dim`，拆成 normal branch 和 overlap branch；token 既以 normal branch 写入本 block 的 compressed entry，也以 overlap branch 写入下一个 block 的 compressed entry。
  - 用 `B0=[t0..t3]`、`B1=[t4..t7]` 的例子说明：`C0 <- normal(B0)`，`C1 <- overlap(B0) + normal(B1)`，压缩后 entry 数仍约为 `n/m`。
  - 压缩权重由可学习的 gate/score 决定，不是简单平均池化。
- 解释 indexer：
  - indexer 有自己的 compressed indexer keys。
  - query 通过低秩投影得到 indexer queries。
  - 用 index score 选 top-k compressed blocks。
- 解释 sliding window：
  - 因为 compressed block 有因果边界，query 不能直接看到同一 block 里的未来/当前细粒度 token。
  - 最近 token 对语言建模很重要，所以保留一段 uncompressed sliding window。
- 解释输出投影：
  - shared KV MQA 降低 KV head 成本。
  - grouped output projection 降低输出投影计算量。
- 必须加入具体例子：
  - 以 DeepSeek-V4-Flash 的 1M context decode 为例，CSA 使用 `compress_ratio=4`、`index_topk=512`、`sliding_window=128`。
  - 说明 1,048,576 tokens -> 262,144 compressed entries -> top-512 compressed entries + 128 raw window -> 640 attention entries。
  - 强调 640 是主 attention 访问的 entries 数，不等同于只保留 640 个原始 token。

建议配图：

- 图 1：CSA 数据流，从 hidden states 到 compressed KV、indexer、top-k、sliding window、attention output。
- 图 2：每 `m` 个 token 压成 1 个 compressed KV entry 的 block 示意图。

AI prompt：

```text
请写“CSA：先压缩 KV，再做稀疏选择”这一节，要求讲得详细但不要堆论文公式。

必须覆盖：
1. CSA 的完整数据流：hidden states -> compressor -> compressed KV entries -> indexer -> top-k selected entries -> concat sliding window KV -> shared KV MQA -> output projection。
2. 解释每 m 个 token 压成 1 个 compressed KV entry，压缩发生在 sequence/block 维度。
3. 解释 learned gated pooling：压缩权重来自模型学习，不是平均池化。
4. 解释 overlapped compression 的机制和直觉：normal branch 写入本 block，overlap branch 写入下一个 block，缓解 block 边界的信息割裂，同时不增加 compressed entry 数量。
5. 解释 lightning indexer 如何为 compressed KV blocks 打分并选择 top-k。
6. 解释 sliding window branch 为什么必要。
7. 用 tensor shape 贯穿说明，例如 H: [n, d]、compressed KV: [n/m, c]。
8. 用一个 1M context 的具体例子串起 compression、indexer、sliding window 和 core attention 的数量变化。
9. 最后总结 CSA 解决了什么：同时减少 KV cache 长度和 attention 访问数量。

代码对应关系可以参考：
- `inference/model.py` 中的 `Compressor`
- `inference/model.py` 中的 `Indexer`
- `inference/model.py` 中的 `Attention.forward`
- `inference/kernel.py` 中的 `sparse_attn`

写作要求：
- 每个公式后都要解释变量含义。
- 不要把 CSA 简化成普通 sparse attention；它是 compression + sparse selection 的组合。
- 不要省略 sliding window，否则因果性和局部细节保留讲不完整。
- 不要在 CSA 流程中插入 MLA 继承关系的长段说明；第 4 节必须按 CSA 数据流推进。
```

## 5. HCA：更高压缩率下的全局注意力

本节目标：把 HCA 和 CSA 的差异讲清楚。HCA 的重点是“压得更狠，并在压缩后的 KV 上保持 dense attention”。

内容要点：

- HCA 的总体流程和 CSA 类似，也有 token-level compressor、shared KV MQA、grouped output projection、sliding window。
- 核心差异：
  - HCA 使用更大的压缩率 `m'`，技术报告中强调 `m' >> m`。
  - HCA 不使用 sparse top-k selector。
  - HCA 在 heavily compressed KV entries 上做 dense attention。
- 直觉解释：
  - CSA 像是在大量压缩块里挑相关块，适合保留较精细的长程信息。
  - HCA 像是把很长历史做成更粗粒度的全局摘要，保证模型始终能看到全局背景。
- 结合配置说明：
  - Flash 配置中 `compress_ratios` 基本在 `4` 和 `128` 之间交替。
  - 可以将 `4` 理解为 CSA 分支的较轻压缩，将 `128` 理解为 HCA 分支的重压缩。
- 必须加入具体例子：
  - 沿用 1,048,576 tokens 的 decode 示例，HCA 使用 `compress_ratio=128`。
  - 说明 1,048,576 tokens -> 8,192 heavily compressed entries，再加 128 sliding window，主 attention 访问 8,320 entries。
  - 强调 HCA 覆盖完整历史，省去 indexer/top-k，但每个 compressed entry 的信息粒度更粗。

建议配图：

- 图 1：HCA 数据流图，突出没有 indexer/top-k，compressed KV 直接进入 dense attention。
- 图 2：CSA 与 HCA 对比图：`m=4 + sparse top-k` vs `m'=128 + dense over compressed entries`。

AI prompt：

```text
请写“HCA：更高压缩率下的全局注意力”这一节。

必须覆盖：
1. HCA 和 CSA 的共同点：都压缩 KV，都使用 shared KV MQA、grouped output projection、sliding window。
2. HCA 的关键差异：压缩率更大 m' >> m，不做 sparse top-k，在压缩后的 KV entries 上做 dense attention。
3. 用 tensor shape 解释：原始长度 n 被压到 n/m'，再在这些 compressed entries 上做 attention。
4. 解释为什么 HCA 适合提供全局粗粒度信息。
5. 结合 `config.json` 中 `compress_ratios` 的 4/128 交替，说明 CSA/HCA 在层配置上的区别。
6. 用同一个 1M context 例子说明 HCA 的数量变化，并与 CSA 的 top-k sparse over compressed entries 区分开。
7. 明确说明 HCA 不是 mHC，mHC 是残差连接机制。

写作要求：
- 不要把 HCA 写成 sparse attention。
- 不要说 HCA “比 CSA 更高级”，而要讲它们的分工不同。
- 结尾自然过渡到“从 MLA 到 CSA/HCA：DeepSeek-V4 保留了什么、改写了什么”。
```

## 6. 从 MLA 到 CSA/HCA：DeepSeek-V4 保留了什么、改写了什么

本节目标：在 CSA/HCA 机制已经讲清楚之后，回答“DeepSeek-V4 是否还保留 MLA 思想”这个问题。不要提前放在第 3 节，否则会引用尚未介绍的 `Compressor`、`Indexer`、overlap 和 HCA。

内容要点：

- 先给结论：
  - Q/O 路径保留 MLA/LoRA 风格的 low-rank projection。
  - 长程 KV cache 主路径从 V2/V3 的 per-token latent KV，转向 CSA/HCA 的 block-level compressed KV entries。
- 对照 V2/V3 MLA：
  - 写出 $c_t^{KV}=W^{DKV}h_t$，再上投影到 K/V。
  - 强调它压缩的是每个 token 的 KV 表示宽度，sequence length 仍是 $n$。
- 对照 DeepSeek-V4 代码：
  - Q 路径：`wq_a -> q_norm -> wq_b`。Flash 是 `4096 -> 1024 -> 64*512`，Pro 是 `7168 -> 1536 -> 128*512`。
  - O 路径：grouped `wo_a -> wo_b`，结合 `o_groups` 和 `o_lora_rank` 说明。
  - KV 路径：sliding-window KV 由 `wkv: hidden_size -> head_dim` 生成，`num_key_value_heads=1`，属于 shared KV / MQA 风格。
  - 长程 KV：CSA/HCA 通过 `Compressor` 沿 sequence/block 聚合为 compressed KV entries，没有 V2/V3 MLA 的 per-token latent KV up-projection 路径。
  - CSA Indexer：复用 Q 的低秩中间态 `qr`，再投到 `index_head_dim=128`，用低维 compressed indexer keys 选 top-k。
- 最后用表格总结：Q projection、O projection、KV cache、CSA indexer 四个位置分别如何继承或改写 MLA 思想。

AI prompt：

```text
请写“从 MLA 到 CSA/HCA：DeepSeek-V4 保留了什么、改写了什么”这一节。

必须覆盖：
1. 本节放在 CSA/HCA 之后，因为读者已经理解 Compressor、Indexer、overlap 和 HCA。
2. 说明 V4 没有简单抛弃 MLA；它保留 Q/O 的 low-rank projection，但长程 KV cache 主路径转向 block-level KV compression。
3. 对照 V2/V3 MLA 的 KV 低秩路径：down projection 得到 $c_t^{KV}$，再 up projection 得到 K/V。
4. 对照 V4 代码解释 Q/O/KV/Indexer 四条路径。
5. 说明 CSA/HCA 的 compressor 投影确实在 hidden_size 维度上降维，但后续没有 MLA 的 per-token K/V up-projection，而是做 block-level learned gated pooling。
6. 用一个表格总结“位置 / V4 中的体现 / 与 V2/V3 MLA 的关系”。

写作要求：
- 不要提前重新解释 CSA/HCA 细节，只引用前文已经讲过的机制。
- 不要用“完全抛弃”或“简单继承”这种二元说法。
- 结尾过渡到“混合注意力架构：CSA 与 HCA 如何分工”。
```

## 7. 混合注意力架构：CSA 与 HCA 如何分工

本节目标：解释 DeepSeek-V4 为什么需要同时使用 CSA 和 HCA。

内容要点：

- CSA 和 HCA 的互补关系：
  - CSA：较轻压缩 + 稀疏选择，兼顾较细粒度的长程检索。
  - HCA：重压缩 + dense compressed attention，提供更便宜的全局覆盖。
- sliding window 是两者共同的局部补偿机制。
- 层间交替可以让模型在不同层中交替处理细粒度相关信息和粗粒度全局信息。
- 结合 `compress_ratios` 说明 DeepSeek-V4-Flash 的配置模式：开头和结尾有不压缩层，中间大体交替使用 4 和 128。
- 用一句话总结：Hybrid Attention 是一套在 sequence compression、sparse selection、local window 和 layer schedule 之间做平衡的系统设计。

AI prompt：

```text
请写“混合注意力架构：CSA 与 HCA 如何分工”这一节。

必须覆盖：
1. 对比 CSA 和 HCA 的分工：CSA 负责较细粒度的长程选择，HCA 负责低成本全局覆盖。
2. 解释为什么只用 CSA 或只用 HCA 都不理想。
3. 说明 sliding window 在两者中都用于补充局部细节。
4. 结合 `compress_ratios` 的层配置，解释 4/128 交替的直觉。
5. 用一张建议图展示不同层中 CSA/HCA 交替出现。

写作要求：
- 标题和正文都用“分工”“互补”“混合架构”这类中性表达。
- 不要使用“旧方案被替代”这类说法。
- 不要把配置数字写死成所有版本完全一致，除非能从对应配置或报告中确认。
```

## 8. 从论文公式到推理代码：CSA/HCA 的实现细节

本节目标：让文章不只是复述论文，而是能对照开源 inference 代码解释工程实现。

内容要点：

- `ModelArgs` / `config.json`：
  - `window_size` / `sliding_window`
  - `compress_ratios`
  - `head_dim`
  - `rope_head_dim`
  - `index_topk`
  - `index_head_dim`
- `Compressor`：
  - `wkv` 生成待压缩 KV。
  - `wgate` 生成压缩权重。
  - `ape` 是 learnable positional bias。
  - `compress_ratio == 4` 时开启 overlap。
  - decode 阶段通过 state buffer 增量维护压缩块。
- `Indexer`：
  - 只在 `compress_ratio == 4` 时出现，对应 CSA。
  - 使用自己的 compressor 生成 indexer keys。
  - 使用 query 和 compressed keys 打分，选 top-k。
- `Attention.forward`：
  - 先生成 query 和当前 window KV。
  - 再生成 window top-k indexes。
  - 如果有压缩分支，再拼接 compressed indexes。
  - 调用 `sparse_attn`。
- `sparse_attn` kernel：
  - 根据 top-k index gather KV。
  - 使用 online softmax 风格计算。
  - 加入 attention sink。
- 低精度细节：
  - RoPE 维度保留较高精度。
  - 非 RoPE 维度可做 FP8/FP4 量化模拟。
  - indexer 中使用 FP4 simulation。

AI prompt：

```text
请写“从论文公式到推理代码：CSA/HCA 的实现细节”这一节。

必须对照以下代码模块解释：
1. `inference/model.py` 的 `Compressor`：解释 wkv、wgate、ape、overlap、kv_state/score_state。
2. `inference/model.py` 的 `Indexer`：解释它如何生成 index score 并选 top-k，为什么主要对应 CSA。
3. `inference/model.py` 的 `Attention.forward`：解释 sliding window KV、compressed KV、topk_idxs 如何汇合。
4. `inference/kernel.py` 的 `sparse_attn`：解释它如何按 top-k indexes gather KV 并做 attention。
5. `config.json` 的 `compress_ratios`、`sliding_window`、`index_topk`、`rope_head_dim` 如何对应论文概念。

写作要求：
- 不要逐行翻译代码，而是按数据流解释。
- 每解释一个类，都要说明它在论文概念中的角色。
- 代码片段只摘关键 5-10 行，避免整段贴代码。
- 需要指出：开源 inference 代码是理解实现细节的依据，论文会省略一些工程细节。
```

## 9. 效果评估：效率收益与长上下文表现

本节目标：用实验结果收束全文，让读者知道 CSA/HCA 带来的实际收益，而不是只停留在结构设计。

内容要点：

- 效率指标：
  - 技术报告中，1M context 下 DeepSeek-V4-Pro 相对 DeepSeek-V3.2 只需要约 27% 的 single-token inference FLOPs 和 10% 的 KV cache。
  - DeepSeek-V4-Flash 相对 DeepSeek-V3.2 只需要约 10% 的 single-token inference FLOPs 和 7% 的 KV cache。
- 解释这些数字时要说明：
  - FLOPs 是 equivalent FP8 FLOPs 口径。
  - 这里比较的是长上下文场景，尤其是 1M token。
  - 收益来自 hybrid CSA/HCA、低精度计算/存储等组合，不要全部归因于单个模块。
- 长上下文能力：
  - 可以引用 MRCR 1M、CorpusQA 1M、LongBench-V2 等相关结果。
  - 重点解释长上下文能力与效率之间的关系：核心在于能以可接受成本使用更长窗口。
- 结尾观点：
  - CSA/HCA 的意义是把百万上下文从“理论上能放进模型”推进到“推理成本更接近可用”。

AI prompt：

```text
请写“效果评估：效率收益与长上下文表现”这一节。

必须覆盖：
1. 引用技术报告中 1M context 下相对 DeepSeek-V3.2 的效率数字：
   - V4-Pro：约 27% single-token inference FLOPs，10% KV cache。
   - V4-Flash：约 10% single-token inference FLOPs，7% KV cache。
2. 说明这些收益来自 hybrid CSA/HCA 与低精度计算/存储等整体设计，不要全部归因于 CSA 或 HCA 单独一个模块。
3. 结合长上下文 benchmark 说明模型不是只省算力，也保持了 1M context 下的可用能力。
4. 最后总结 CSA/HCA 对长上下文模型设计的启发。

写作要求：
- 所有数字必须来自技术报告或 README，不要编造。
- 解释图表时先说结论，再说图里的证据。
- 不要把 benchmark 写成排行榜，要回到本文主线：长上下文效率。
```

## 可直接交给 AI 的总 prompt

```text
请根据以下资料，写一篇中文技术文章/报告：《DeepSeek-V4 的百万上下文注意力设计：CSA 与 HCA》。

资料来源：
1. DeepSeek-V4 技术报告：/Users/bowenyuchi/Projects/DeepSeek-V4/DeepSeek-V4-Flash/DeepSeek_V4.pdf
2. 如果本地 PDF 不可读，使用官方链接：https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
3. 推理代码目录：/Users/bowenyuchi/Projects/DeepSeek-V4/DeepSeek-V4-Flash/inference
4. 配置文件：/Users/bowenyuchi/Projects/DeepSeek-V4/DeepSeek-V4-Flash/config.json

文章目标：
- 面向有一定 LLM/Transformer 基础的工程师做技术分享。
- 从 DeepSeek-V4 的基本情况切入，再进入长上下文 attention 瓶颈，最后详细解释 CSA/HCA。
- 语言要自然、平实、图文并茂，避免 AI 味道。
- 既要讲论文原理，也要对照 inference 代码解释实现细节。

文章结构：
1. 认识 DeepSeek-V4
2. 长上下文的核心瓶颈：KV cache 与 attention 计算
3. Attention 压缩路线：从 MHA 到 DeepSeek MLA/DSA
4. CSA：先压缩 KV，再做稀疏选择
5. HCA：更高压缩率下的全局注意力
6. 从 MLA 到 CSA/HCA：DeepSeek-V4 保留了什么、改写了什么
7. 混合注意力架构：CSA 与 HCA 如何分工
8. 从论文公式到推理代码：CSA/HCA 的实现细节
9. 效果评估：效率收益与长上下文表现

关键要求：
1. 不要把 HCA 和 mHC 混淆。HCA 是注意力机制，mHC 是残差连接增强机制。
2. 不要把 MHA、MLA、DSA 称为“旧路线”或“过时方案”，要写成一条 attention 压缩技术的演进路线。
3. 讲 MHA/MLA/DSA 时必须说明它们分别压缩哪个 tensor 维度：
   - head/KV head 维度
   - hidden/latent 表示维度
   - sequence/block 访问维度
4. 讲 CSA 时必须覆盖 compression、lightning indexer、top-k sparse selection、sliding window、shared KV MQA、grouped output projection。
5. 讲 HCA 时必须强调它使用更高压缩率，不做 sparse top-k，而是在 heavily compressed KV entries 上做 dense attention。
6. 讲 V4 与 MLA 的关系时必须放在 CSA/HCA 之后，说明 Q/O 低秩投影保留，KV cache 主路径转向 block-level compression。
7. 讲 Hybrid Attention 时必须解释 CSA/HCA 的分工和互补关系。
8. 讲实现时必须对照 `Compressor`、`Indexer`、`Attention.forward`、`sparse_attn`、`compress_ratios`。
9. 讲实验时必须引用报告中的 1M context 效率数据：V4-Pro 相对 V3.2 为约 27% FLOPs 和 10% KV cache，V4-Flash 为约 10% FLOPs 和 7% KV cache。

风格要求：
- 每节开头先给直觉，再进入公式或代码。
- 所有公式后都要解释变量和直觉。
- 多使用“换句话说”“从 tensor 角度看”“工程上对应到”这类自然衔接。
- 少用排比口号，不要使用营销式形容词。
- 图表建议用文字描述出来，例如“这里建议画一张……图”。
```
