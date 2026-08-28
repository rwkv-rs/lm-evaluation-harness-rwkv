# 发布评测结果到 scoreboard-rwkv

评测完成并保存标准 lm-eval 结果后，`lm_eval.loggers.scoreboard` 在同一进程内完成：

1. 将聚合指标、逐样本证据和 provenance 转换成 `scoreboard-v1`。
2. 写入 `publication/raw_results.json`、`campaign.json`、`tasks/*.json` 和
   `status.json`。
3. 调用 preflight → campaign → task → finalize。

没有独立转换 CLI，也没有第二套上传格式。网络或契约失败不会删除原始结果；
`status.json` 会记录 `evaluation=complete, publication=failed`。

## 配置与凭据

```bash
export SCOREBOARD_PUBLICATION_TOKEN=local-publication-token
```

上传器不读取默认 endpoint、模型或重试环境变量。base URL、token 变量名、模型 SHA、
revision、timeout 和重试策略都由 publication 配置显式提供；只有 token 值通过配置指定的
环境变量读取，避免密钥进入配置或结果。

## Comparison 配置

标准 lm-eval YAML 通过 `metadata.scoreboard_publication` 提供显式 producer contract。
支持 `publication` 字段的版本化配置可以直接传入相同对象；其中既有的
`publication.task_metadata` 与 `publication.tasks` 等价。

```yaml
metadata:
  scoreboard_publication:
    base_url: http://127.0.0.1:7862
    token_env: SCOREBOARD_PUBLICATION_TOKEN
    model_sha256: <实际加载权重的 64 位小写 SHA-256>
    model_revision: <实际权重 revision>
    timeout: 3600
    control_timeout: 30
    retries: 2
    retry_delay: 1
    tasks:
      gsm8k_platinum:
        outcome_metric: exact_match
        response_mode: single
        comparison:
          model: &model
            label: RWKV7 G1I 1.5B
            architecture: RWKV7
            generation: G1I
            parameters: 1.5B
          benchmark:
            label: GSM8K Platinum
            categories:
              - id: math
                label: 数学
            evaluation_method: exact_match
            score_multiplier: 100.0
          evaluation:
            prompt_profile: fake_think
            prompt_template: "User: {task.problem}\n\nAssistant: <think></think>"
            precision: fp32io16
          coordinates:
            - comparison:
                id: precision
                label: fp16 vs fp32io16
                short_label: 精度
                a_label: fp16
                b_label: fp32io16
                contract: 同一权重、prompt、sampling 与输出边界，仅改变 WKV precision。
              parameter_group:
                id: 1.5b
                label: 1.5B
                a_model: *model
                b_model: *model
                parameter_delta_percent: 0.0
                comparable: true
              arm: b
```

task policy 可使用完整 task name 或 task metadata 中的 `benchmark_name`。以下字段必须由
配置明确提供，上传器不会从权重名或 tags 猜测：

- 当前模型的架构、代际和参数组。
- comparison ID、A/B 标签、contract、参数组和当前 arm。
- benchmark 展示名、分类、evaluation method 和 score multiplier。
- prompt profile、prompt template、precision 和用于判定正确性的 `outcome_metric`。
- response mode：单次生成使用 `single`，选择题 likelihood 使用 `multiple_choice`。

以下字段只使用真实评测结果计算，配置值会被覆盖：

- `comparison.samples`
- `comparison.truncation_rate`
- 每条样本的 `answer.outcome`、标准答案、提取答案、prompt、completion、失败原因、
  token 数和 latency（若后端提供）

`outcome_metric` 的逐样本值为 `1` 时记录 `correct`，为 `0` 时记录
`incorrect`；缺失或非二值指标记录为 `unanswered`/`undetermined`，不会伪造判断。

## 严格失败边界

- 每个 task 都必须有显式 policy；缺失时拒绝发布。
- 只有显式设置 `history_only: true` 才允许不提供 `comparison`，该结果只进入分数历史。
- comparison publication 必须为每条样本生成完整 `answer`。
- 缺少真实模型 SHA-256、WKV mode、聚合指标、原始 response 或 input/output token IDs
  时拒绝发布。
- 已完成 campaign 内容不可改变；正式重跑必须提供真实且不同的 `rerun_reason`。
