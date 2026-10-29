## 核心目标
lm-eval-harness 是 LLM 社区的标准评估库, 很早成为最主流的评估框架, 本仓库需要按照社区主流做法完成 RWKV 模型的接入.
本仓库只需要用于完成 lighteval 和 evalscope **未能收录** 的 benchmark 的评估.

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
1. 显存余量应当小于总量的 10%, GPU 利用率应达到 97%, 如 transformers-rwkv 或 FlashRWKV2 存在性能问题, 请及时反馈给用户.
2. 使用 Http 协议支持评估端与推理端的通信, 请求并发量应当略大于推理并发量(允许少量排队, 但禁止空载)

## 结果保存
记录详细 (benchmark_name, model_name, n_samples, k_metrics, cot_mode, prompt_template), [_可选完成 wkv_mode fp32io16 vs fp16 对比] 对应的 (正确率, 截断率) , 其中截断率定义为达到输出上限未能完成作答的样本数 / 总样本数

## 分数上传
 scoreboard-rwkv 仓库承担，在配置了url和token之后，即可发布

## Env
使用 uv 管理本机和远端专属环境 ./.venv, 严禁本项目使用其它环境, 严禁其它项目使用本项目环境, 避免环境污染问题。


use git to sync your changes instead of rsync.
