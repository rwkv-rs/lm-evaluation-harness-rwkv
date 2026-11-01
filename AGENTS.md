## 核心目标
本仓库基于 EleutherAI/lm-evaluation-harness，通过原生的 `simple_evaluate()` / `evaluate()` 评测流水线、`LM` / `TemplateLM` 模型接口和 `Task` / `ConfigurableTask` 任务接口，完成 RWKV HTTP 模型的集成。
本仓库的 RWKV 评测范围限于 LightEval 和 EvalScope **未能收录**、且能够通过本仓库任务接口执行的 Benchmark。新增或调整评测范围前，必须核对这两个框架当前可运行的任务注册表；保留上游原生任务不代表将其全部纳入 RWKV 生产评测。


## 代码原则：

对于每一个文件、类型、函数和变量，都必须先找到一个相似的现有实现作为原型（Prototype）。
- 如果原型的名称中包含模型名称，则将该模型名称替换为 RWKV 或相应的大小写变体。
- 如果原型的名称中不包含模型名称，则保持原有名称不变。
- 应通过添加条件分支来实现新功能，不得影响现有功能。
非干扰原则：

我们引入的修改不得破坏 lm-eval-harness原生的评测流水线，也不得影响其他模型的评测结果。

## 评测生产端契约
1. 任何可能影响分数的修改都必须获得用户明确批准
在进行任何可能影响评测分数的修改之前，必须获得用户的明确批准，包括：
- 评分规则；
- 答案抽取或判分修复；
- 期望分数处理；
- 指标聚合；
- Prompt 或评测模式；
- 停止行为；
- Token 预算。
用户明确提出某项具体修改，即视为对该项修改的批准。绝不能将代码清理或重构视为对评分行为修改的授权。

已经批准的 RWKV 缺失选项期望分数处理方式（0.25，以及历史使用的 pass@1 缺失 rollout 公式），只能保留在 RWKV 私有判分器中。

部分分数（Fractional Credit）不能使答案抽取失败的评测明细变成通过状态。
2. 评测模式必须对应实际执行行为
评测模式是实际执行的行为，而不是一个标签。

所有 NoCoT（无思维链）选择题任务，都必须通过 /v1/completions 接口，使用原生的 LOGPROBS / loglikelihood（对数似然）评分。

对于 NoCoT 选择题，绝不能用生成式精确匹配（Generative Exact-Match）代替原生 loglikelihood 评分。

选择题必须正确接入 lm-eval-harness 原生任务和判分接口：
- `output_type: multiple_choice`；
- `doc_to_text`、`doc_to_choice` 和 `doc_to_target`；
- `Task.construct_requests()` / `ConfigurableTask.construct_requests()` 生成的 `Instance(request_type="loglikelihood")`；
- 模型的 `LM.loglikelihood()`，以及任务的 `process_results()`、`metric_list` 和 `aggregation`。
原生生成任务使用 `output_type: generate_until` 和 `LM.generate_until()`；是否使用思考模式，应单独记录。
`generation_kwargs.max_gen_toks`、模式标签或元数据字段不能代替实际的 `Instance.request_type` 和 HTTP 请求证据。
3. Benchmark 原生任务配置具有最高权威性
Benchmark 自身的原生任务配置是最终依据。

自动模式选择可以选择：
- NoCoT
- FakeCoT
- CoT
但在发布结果之前，必须根据实际流水线日志及 HTTP 请求、响应证据，验证执行的是 `loglikelihood`（真实 LOGPROBS）还是 `generate_until`（GENERATIVE）。仅记录适配器入口方法名不能证明内部实际执行的请求类型。

即使上传的模式标签看起来正确，只要实际评测模式不匹配，该分数就无效。
4. 评测明细中的 Prompt 必须与实际输入一致
NoCoT 的评测明细必须反映实际发送给 Completion 接口的输入，并且不能包含 CoT Prompt。

CoT / FakeCoT 的评测明细必须保存应用已配置的 RWKV Prompt Template 后，模型实际接收的 Prompt：
- `/v1/completions` 由客户端渲染时，记录实际发送的完整 Prompt；发送 Token ID 时，还须保留对应 Token ID 与准确的文本输入证据；
- `/v1/chat/completions` 由服务器渲染时，必须捕获服务器实际渲染的 Prompt。

严禁：
- 在上传或展示层手动追加选择题选项；
- 重复添加 Answer:；
- 将内部候选请求的 JSON 封装暴露为用户消息；
- 为展示用途重新构造一个与实际输入不同的 Prompt。
如果生成式评测未捕获上述对应路径的实际输入证据，则该结果不能发布。

不得在这种情况下悄悄上传原始消息列表来代替实际 Prompt。
5. CoT / FakeCoT 的停止条件和答案抽取
CoT / FakeCoT 生成不能继承原生任务中用于单个答案的换行停止序列。

换行符可以作为单 Token 补全任务的停止条件，但对于推理过程，它会在模型生成最终答案之前截断推理轨迹。

思考模式请求必须允许模型通过自身的停止协议或 EOS 机制完成生成。

在计算指标之前，RWKV 的生成输出必须移除：
- 推理包装内容；
- 任何 Assistant 消息包装内容。
未正常闭合的 <think> 推理轨迹必须判定为失败或截断，而不能被当作有效答案。

对于本仓库评测范围内需要数学答案抽取的 RWKV 任务，必须使用 Albatross 风格的简洁数学答案抽取器：
- 优先提取括号平衡的 \\boxed{...}；
- 如果无法提取，则回退到最终的数学表达式。
对于本仓库评测范围内以代码围栏（Fenced Code Block）输出程序的 RWKV 任务，必须按经批准的任务抽取规则提取最终程序，再上传评测明细。
上述要求不构成新增 Math、MATH-500 或 LiveCodeBench 的授权；任务是否纳入本仓库，仍须满足评测范围与接入审查要求。

Scoreboard 评测明细中的 answer 和 ground_truth 必须保存这些简洁的答案值，而不是完整的 Assistant 消息或完整的解题过程。
6. 答案抽取失败的表示方式
答案抽取失败时，内部判分逻辑必须保留 None。

但上传到 Scoreboard 的 answer 字段必须使用字符串：

"-"

绝不能使用：
- JSON null
- 字符串 "None"
必须保留原始 Assistant 响应及失败分类。

用于显示的占位符不能被当作有效答案，也不能改变最终分数。
7. RWKV 数学验证器的超时与回退规则
已经批准的 RWKV 数学验证器，对解析和等价性验证设置统一的 5 秒墙钟时间上限（Wall-Clock Deadline）。

如果验证无法在规定时间内完成，则回退到以下判断方式：

对抽取出的标准答案和预测答案分别去除首尾空白，然后进行完整字符串相等比较。

两个字符串不同，则分数为 0。

绝不能使用子字符串匹配。

符号计算必须运行在可终止、具有内存限制的 Worker 中，以保证：
- 调用方线程不能绕过或使超时限制失效；
- 巨大整数展开不能耗尽宿主机资源。
同时必须保持以下内容不变：
- 答案抽取；
- 指标聚合；
- Prompt；
- 生成 Token 预算；
- 所有非 RWKV 模型的行为。
8. 多 Token 选项的 Loglikelihood 评分
对于由多个 Token 组成的候选选项，必须将该候选序列中每个 Token 的条件对数概率相加，作为最终分数。

这属于序列对数似然（Sequence Loglikelihood），不是困惑度（PPL）。

对于单 Token 选择题，在服务器能够提供可靠的指定 Token Prompt Logprobs 之前，也可以继续使用相同的原生评分路径。

本仓库评测范围内的所有 NoCoT 选择题，必须使用原生 `output_type: multiple_choice`，由 `doc_to_text`、`doc_to_choice`、`doc_to_target` 和 `construct_requests()` 构建真实的 loglikelihood 请求，再由原生 `process_results()` 及配置的指标判分。
其 Prompt 应遵循任务原生配置及已批准的 RWKV 格式要求，不能加入推理指令，也不能附带生成式答案包装内容。

所有 RWKV NoCoT + LOGPROBS Prompt 都必须以独立换行的 \nAnswer: 提示符结束。

如果已有该提示符，必须保留其原有内容及空白字符。

如果不存在，则应在 RWKV 私有 Prompt 格式化逻辑中追加：
`\nAnswer: `


不能因此修改候选选项、标准答案或评测指标。

只有生成式 Prompt 才应移除末尾的 Answer:。
9. 显式配置生成 Token 预算
必须针对每个模型和任务明确配置生成 Token 预算：
- 普通生成任务：最多 4096 Token；
- 高难度数学和推理任务：最多 24576 Token。
例如：

4096 / 8192 / 16384 / 24576

这些数值表示的是生成补全长度（Completion Length），不是总上下文长度。

发送请求时，必须原样传递这些预算值，不能减去 Prompt 所占用的 Token 数量。

推理后端具有单独配置的、足够大的上下文容量。

不能从生成预算推断推理后端的总上下文限制，也不需要强制进行预先 Tokenization 检查。

每条评测结果都必须记录：
- 模型；
- 实际请求的生成 Token 预算；
- 声明的上下文元数据；
- Prompt Template；
- 请求类型；
- 截断率。
不能将 Manifest 或模型权重中的上下文元数据，当作实际测量得到的服务器上下文上限。

如果生成达到输出长度上限，但尚未完成有效答案，则必须将其视为截断，而不是有效答案。
10. 保持评测生产端代码、配置及运行生命周期一致
必须确保评测生产端的源码、配置、入口程序和实际运行生命周期保持一致。

在重新执行评测之前，必须检查：
- PID 对应的启动参数（argv）；
- cgroup；
- systemd Unit。
只能停止经过确认、确实负责受影响评测结果的 Producer 任务。

严禁停止：
- 推理服务；
- Gateway；
- 无关的评测任务。
也不得运行已删除的旧版 CLI 或配置文件。
11. 评测结果修正流程
当需要修正评测结果时，必须保留现有数据库记录及审计明细。

修正流程如下：
1. 获取最新的生产环境 Manifest。
2. 使用临时迁移脚本，仅将精确识别出的受影响 Score ID 标记为过期。
3. 从本仓库配置重新解析受影响的 task / group / tag，并核对展开后的叶子任务与 checkpoint 身份。
4. 将重跑范围限制在受影响的模型与 Benchmark 组合。
5. 上传之后验证替换结果的 ID、模式、请求类型、Token 预算和截断率。
严禁为了修正结果而截断或清空 Scoreboard 数据库。

不得将不相关 Benchmark 的结果标记为过期。
12. Smoke Test 与生产运行要求
冒烟测试（Smoke Validation）必须使用规定的 1.5B 和 2.9B 模型样本，证明实际发出了对应的 LOGPROBS 与 GENERATIVE 请求。

生产评测运行必须：
- 流式保存已完成的 Rollout；
- 支持中断与恢复；
- 将凭据限制为仅 root 可访问，或仅通过环境变量提供。
不得将凭据保存在源码、日志或命令行参数（argv）中。
13. 严格划分职责边界
本仓库负责：
- lm-eval-harness 评测生产流水线；
- RWKV HTTP 适配器；
- Scoreboard 结果发布。
外部系统负责：
- 推理服务生命周期；
- 模型权重；
- wkv mode；
- Endpoint Manifest。
仅用于评测的临时 Unit，可以在一次性运行中复用现有生命周期操作，但不能演变成永久调度器。

不得从本仓库修改 Scoreboard 的 AGENTS.md 或其实现代码。




## 强制 Benchmark 接入审查
在新增 Benchmark，或者修改已有 Benchmark 的以下内容之前：
- Prompt（`doc_to_text`、few-shot、`gen_prefix`、chat template）；
- 请求构建（`output_type`、`construct_requests()`、`Instance.request_type`）；
- 判分与聚合（`process_results()`、`metric_list`、`aggregation`、group 聚合）；
- Benchmark 分类（`configs/benchmarks.toml` 中的 field）；
- CoT 模式、答案格式、`filter_list` / 答案抽取器；
- Scoreboard 上传契约；
必须完整阅读本仓库现有文档：

- [新增任务指南](docs/new_task_guide.md)；
- [任务、过滤器与指标配置指南](docs/task_guide.md)；
- [模型接口指南](docs/model_guide.md)；
- 涉及生产配置、恢复或发布时，还须阅读 [配置指南](docs/config_files.md) 和 [RWKV 评测指南](RWKV_EVALUATION_GUIDE.md) 的相关章节。

还必须阅读本文件中的“评测生产端契约”“正确性检查”以及本节的端到端验收要求。仓库指南中的历史说明与本文件不一致时，必须指出差异，不能以旧说明豁免本文件的评测契约。

固定 Benchmark 接入工作流程：

1. 核对 Benchmark 官方协议、lm-eval-harness 原生任务，以及 LightEval / EvalScope 的收录情况，确认本仓库负责的范围。
2. 找到已有 task YAML、模型适配器、过滤器、判分器及测试作为原型，记录原型路径、复用方式与修改边界。
3. 记录数据集版本、split、预处理、Prompt、候选项、标准答案、请求类型、采样、停止、Token 预算、抽取、指标、聚合及上传契约；任何可能影响分数的修改须先获得用户明确批准。
4. 沿 `TaskManager` → 原生 Task → `construct_requests()` / `Instance` → HTTP 模型适配器 → `filter_list` → `process_results()` / 聚合 → 结果工件与 Scoreboard 的实际链路实现，RWKV 专属行为必须隔离。
5. 完成隔离测试和下述端到端验收，核对实际 HTTP 请求、结果工件与测试榜单记录，随后才能审查生产配置准入。

Benchmark 修改的最低审查条件：
一项 Benchmark 修改，必须至少证明以下内容，才具备可审查性：
1. 数据集字段、版本与 split，`process_docs`、`doc_to_text`、`doc_to_choice`、`doc_to_target` 的实际输出、标准答案格式，以及 `TaskManager` 展开 task / group / tag 后的叶子任务清单。
2. Benchmark 原生任务协议与实际 NoCoT / FakeCoT / CoT 模式，`output_type`、`Instance.request_type`、`Instance.args`、生成采样参数和 `repeats` / k；这些字段必须与实际执行一致。
3. 通过日志验证实际使用的请求路径是 /v1/completions 还是 /v1/chat/completions。
4. Prompt Template、停止行为、生成 Token 预算、答案抽取器、抽取失败值和截断语义。
5. 原生 `filter_list`、`process_results()`、`metric_list`、任务与 group 聚合的接线方式，评测明细中使用的实际模型输入，以及 answer、ground_truth、is_passed 行为的一致性。
6. Hermetic Tests（隔离且可复现的测试），以及发布前必须进行的 1.5B 和 2.9B 模型各 10 道题冒烟评测。
7. 测试数据库中必须存在受影响 Benchmark 的实际分数与明细记录，并且此前报告的答案抽取、判分、Prompt 和模式异常均已消除。
仅仅启动 Producer、通过单元测试，或者查询得到一个空结果集，均不能视为验收通过。

不能仅根据 cot_mode 标签、`generation_kwargs.max_gen_toks` 或任务名称判断真实的评测执行行为。

不得在上传层手动重构 NoCoT 选项或已经渲染的 CoT Prompt。

如果所需测试文件不存在，在新增文件之前必须获得用户确认。

在将 Benchmark 加入生产环境配置之前，代码审查必须明确确认已经通过此准入审查。
## 目录规范

```text
.
├── .github/
│   └── workflows/                 # GitHub Actions 工作流
├── .gitignore                     # Git 忽略规则
├── .pre-commit-config.yaml        # pre-commit 检查配置
├── docs/                          # lm-eval 使用、模型接入与任务开发文档
│   └── img/                       # 文档图片
├── examples/                      # Notebook 与独立调用示例
├── lm_eval/                       # lm-evaluation-harness 主 Python 包
│   ├── _cli/                      # `lm-eval` 命令行入口与子命令
│   ├── api/                       # Model、Task、Instance、Metric 等核心接口
│   ├── caching/                   # 请求缓存
│   ├── config/                    # 评估与任务配置对象
│   ├── decontamination/           # 数据去污染工具
│   ├── filters/                   # 输出抽取、转换与选择过滤器
│   ├── loggers/                   # 评估结果记录与外部日志集成
│   ├── models/                    # Hugging Face、HTTP/API 等模型后端适配
│   ├── prompts/                   # Prompt 辅助实现
│   ├── tasks/                     # Benchmark 定义与注册
│   │   ├── include/               # 可复用 YAML 片段
│   │   ├── <benchmark>/           # 每个 benchmark 或 benchmark 组的独立目录
│   │   ├── _factory.py            # Task 工厂
│   │   ├── _index.py              # Task 索引
│   │   ├── _yaml_loader.py        # YAML Task 加载器
│   │   └── manager.py             # TaskManager 实现
│   ├── __init__.py                # 包初始化与版本信息
│   ├── __main__.py                # `python -m lm_eval` 入口
│   ├── defaults.py                # 全局默认评估配置
│   ├── evaluator.py               # 评估主流程
│   ├── evaluator_utils.py         # 评估流程辅助函数
│   ├── result_schema.py           # 结果数据结构
│   └── utils.py                   # 通用工具函数
├── scripts/                       # 构建 benchmark、回归比较与结果处理脚本
│   └── clean_training_data/       # 训练数据清理脚本
├── templates/
│   └── new_yaml_task/             # 新建 YAML Task 的模板
├── tests/                         # 单元测试与测试资源
│   ├── models/                    # 模型后端测试
│   ├── scripts/                   # scripts/ 对应测试
│   ├── test_configs/              # 测试配置
│   ├── testconfigs/               # 配置解析测试资源
│   ├── testdata/                  # 测试数据
│   ├── testyamls/                 # YAML Task 测试资源
│   └── test_*.py                  # 核心 API、Task、Evaluator 等单元测试
├── temp/                          # **启动脚本**
│   └── run_<benchmark_name>_<model_name>.sh  # 启动脚本
├── AGENTS.md                      # 本仓库 Agent 约束与项目规范
├── CITATION.bib                   # 项目引用信息
├── CODEOWNERS                     # 代码所有者配置
├── LICENSE.md                     # 项目许可证
├── MANIFEST.in                    # Python 源码包文件清单
├── README.md                      # 项目说明
├── ignore.txt                     # 数据去污染忽略项
├── pile_statistics.json           # Pile 去污染统计数据
└── pyproject.toml                 # 包元数据、依赖、工具与命令入口配置
```

model_name 需要写清楚 Qwen(如 Qwen3.5-2B ) / RWKV7 (详情见 `RWKV7 权重` 一章节) 权重具体版本号.
新增任何文件, 都需要得到用户确认.

## 权威 RWKV7 实现
(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- 权威底层推理引擎实现仓库 (cuda, for pro6000, 无调度, 无varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- 权威预训练实现仓库 (cuda, for h100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Diagonal Plus Low Rank(DPLR）的数学原理：显式转移矩阵的并行计算
(6) https://github.com/rwkv-rs/transformers-rwkv/tree/rwkv -- 权威 RWKV Huggingface Transformers 适配仓库 (with rust tokenizer, x10 faster than python implementation)

## RWKV7 权重
权重一般命名规范: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
如: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: 架构版本, 如 rwkv7(default), rwkv7a(experimental, rwkv7 with DeepEmbed), rwkv7b(experimental, rwkv7 with DeepEmbedAttn)
data_version: 数据版本, 如 g1a, g1b... (The further back in the alphabet, the better)
param_size: 参数规模, 仅有 0.1b, 0.4b, 1.5b(often used in RL), 2.9b, 7.2b(often used in the infer test), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- 权威权重 Release 源 (update every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- 权威权重 Test 源 (不定期update)
(3) https://huggingface.co/rwkv-rs/rwkv7-g1-st -- 权威权重 Release 源 (for transformers)

## 正确性检查
1. 是否能够正确应用 transformers-rwkv 以及对应 rwkv7-g1-st 权重仓库中提供的三组 Prompt Template
2. 默认使用 wkv_mode=fp32io16
3. 当使用 Open Think 模式时, 使用解码参数 temp 0.96, top_p 0.76, top_k 32, presence_penalty 1.0, frequency_penalty 0.1, penalty_decay 0.988; 使用 Fake Think 模式时, 使用解码参数 temperature 1.0, top_p 0.28, top_k 32
4. 参考 https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_gpqa_diamond.py 完成选择题的通用判分器实现
5. 参考 https://github.com/BlinkDL/Albatross/blob/main/faster3a_2605/eval_math500.py 完成简答题的通用判分器实现
6. 模型分数应当于 Qwen3.5 相似参数量模型有相似的得分

## 吞吐量检查
1. GPU 显存和利用率
显存剩余空间应小于总显存的 10%，GPU 利用率应达到 97%。

如果 transformers-rwkv 或 FlashRWKV2 存在性能问题，应及时向用户报告。
2. HTTP 通信与多模型并发
评测端与推理端之间必须使用 HTTP 协议通信。
请求并发量应略大于推理并发量，允许少量排队，但禁止推理资源空闲。
未经用户明确批准，不得为了绕过 HTTP 传输错误而降低 Producer 并发量。
必须在保持要求的 Pool 容量不变的前提下，排查传输瓶颈。
全部四个模型 Pool 必须在独立进程中并发运行，并分别拥有：
- 独立的 Benchmark 队列；
- 独立的故障处理与恢复机制。
绝不能串行评测多个模型，也不能设置跨模型的 Benchmark 同步屏障。
生产环境中的临时 Unit 必须能够独立重启每个模型 Pool。
必须检查每个 Pool 的：
- 实际运行中的请求数量；
- 实际等待中的请求数量；
- Token 吞吐量变化。
仅仅配置了并发参数，或者 Producer 进程正在运行，并不能证明推理端已经达到饱和状态。
3. 使用社区主流 Benchmark 评测方法
必须参考社区中广泛采用的 Benchmark 评测方法，确保正确使用：
- /v1/completions
- /v1/chat/completions
同时确保使用适当的思考强度：
- NoCoT
- FakeCoT
- CoT
以及对应的 Prompt Template，例如：
Question: ... Answer:
或者：
User: ... Assistant: <think


等。

## 中断与恢复
评测任务必须以支持后台运行和自动重启的方式启动。
如果由于需求变更或 Bug 修复而需要重新启动评测，应优先在 /tmp 中创建临时迁移脚本，以复用已有结果，而不是从头重新运行全部评测。
已经完成的 Rollout 必须立即写入磁盘。
发生异常时，应确保只有当前正在执行的 Rollout 丢失。
恢复执行时，只补齐缺失的部分。
任务结束后必须进行清理，避免遗留用户服务等残余资源。
## 结果保存
记录详细 (benchmark_name, model_name, n_samples, k_metrics, cot_mode, prompt_template), [_可选完成 wkv_mode fp32io16 vs fp16 对比] 对应的 (正确率, 截断率) , 其中截断率定义为达到输出上限未能完成作答的样本数 / 总样本数

## 职责边界
本仓库独立负责：
1. 本仓库评测范围内已注册 Benchmark 的 lm-eval-harness 原生评测流水线；
2. RWKV HTTP 模型适配器；
3. 标准评测结果和明细输出；
4. Scoreboard 分数发布。
外部系统负责：
1. 推理服务生命周期；
2. 模型权重切换；
3. wkv_mode 切换；
4. Endpoint Pool Manifest。
尚未接入本仓库任务接口的 Benchmark，须先完成本文件规定的接入审查；已由 LightEval 或 EvalScope 收录的 Benchmark，由对应框架负责。
本仓库不承担跨框架调度，也不管理外部评测任务的生命周期。

## 同步上游
本仓库需要定期同步 lm-evaluation-harness 上游。框架上游为 https://github.com/EleutherAI/lm-evaluation-harness；若使用 RWKV fork 作为协作上游，必须明确记录同步仓库、分支与提交。
同步前必须核对 Git remote 的实际 URL 和目标分支，不能仅凭 remote 名称判断上游身份。LightEval 和 EvalScope 的代码不作为本仓库的框架同步来源。

同步之前，必须先根据以下原则重构代码：
- 对应原则（Correspondence Principle）；
- 非干扰原则（Non-Interference Principle）。
同时完成：
- 移除仅使用一次的 Wrapper；
- 清理过时的静态测试。
随后再同步上游代码。

最终同步状态应显示：
ahead x, behind 0


即本地分支可以领先本次明确选定的上游目标分支若干提交，但不能落后；必须基于最新获取的目标引用核对，并报告比较的仓库、分支及提交。

## 分数上传
 scoreboard-rwkv 仓库承担，在配置了url和token之后，即可发布

## Env
使用 uv 管理本机和远端专属环境 ./.venv, 严禁本项目使用其它环境, 严禁其它项目使用本项目环境, 避免环境污染问题。


use git to sync your changes instead of rsync.
