## LLaMA-Factory 使用说明

此文档支持 OpenAI 格式以及带有多个工具调用的 fc 数据。

### 克隆仓库

```bash
# 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 切换到指定 commit
git checkout d891bfedad9b3f3ca439af0180bdb6c0b1cfd4ab

# 安装
pip install -e ".[torch,metrics]" --no-build-isolation
```

### 配置说明

template使用 `bailing_v2`

llama factory中数据集参数

在 `data/dataset_info.json` 中添加如下数据集配置, 按说明修改相应参数, 有默认值的则无需修改

其他 `sharegpt` 格式数据可以修改 `formatting` 为 `openai` 来兼容Ling模型训练

```json
"数据集名称": {
  "file_name": "该目录下数据集文件夹或文件的名称",
  "formatting": "openai",
  "ranking": "是否为偏好数据集（可选，默认：False）",
  "subset": "数据集子集的名称（可选，默认：None）",
  "split": "所使用的数据集切分（可选，默认：train）",
  "num_samples": "该数据集所使用的样本数量。（可选，默认：None）",
  "columns": {
    "messages": "messages",
    "tools": "tools",
    "images": "数据集代表图像输入的表头名称（默认：None）",
    "videos": "数据集代表视频输入的表头名称（默认：None）",
    "audios": "数据集代表音频输入的表头名称（默认：None）",
    "chosen": "数据集代表更优回答的表头名称（默认：None）",
    "rejected": "数据集代表更差回答的表头名称（默认：None）",
    "kto_tag": "数据集代表 KTO 标签的表头名称（默认：None）"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "observation_tag": "tool",  # 和数据中tool response role对应
    "function_tag": "function_call",  # 和数据中tool calls role对应
    "system_tag": "system"
  }
}
```

### SFT示例
我们使用 [glaive_toolcall_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_en_demo.json) 数据来演示如何如何微调Ling模型，你需要在 [dataset_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json) 文件中将`glaive_toolcall_en_demo`的formatting设置改为`openai`

我们提供了一个对应的demo配置来开启Ling的SFT训练

`llamafactory-cli train examples/sft/llama_factory/ling_full_sft.yaml`

### 数据格式

单条数据示例

```json
{
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "网络搜索",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "query"
                    ]
                }
            }
        }
    ],
    "messages": [
        {
            "role": "system",
            "content": "default system"
        },
        {
            "role": "user",
            "content": "查一下上海天气和最新AI新闻"
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "上海"}
                    },
                    "type": "function",
                    "index": 0
                },
                {
                    "function": {
                        "name": "search_web",
                        "arguments": {"query": "最新AI新闻"}
                    },
                    "type": "function",
                    "index": 1
                }
            ]
        },
        {
            "role": "tool",
            "content": "{\"result\": \"28°C, 多云\"}",
            "tool_call_id": 0
        },
        {
            "role": "tool",
            "content": "{\"result\": \"2025年GPT-5发布在即...\"}",
            "tool_call_id": 1
        },
        {
            "role": "assistant",
            "content": "上海当前天气：28°C，多云。最新AI新闻：2025年GPT-5发布在即，引发行业热议。",
            "reasoning_content": "",
            "tool_calls": []
        }
    ]
}
```

数据附加说明版

```python
{
    "tools": [  # tools字段支持list或json dumps后的str, 不使用tools时为None
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "location"
                    ]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "网络搜索",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "query"
                    ]
                }
            }
        }
    ],
    "messages": [
        {
            "role": "system",  # 没有system可以不加
            "content": "default system"
        },
        {
            "role": "user",
            "content": "查一下上海天气和最新AI新闻"
        },
        {  
            # 工具调用必须按如下格式 role和dataset配置中function_tag(或assistant_tag)对应
            # content为空, tool_calls为list且每一项包含`function`字段为具体的工具调用信息
            "role": "assistant",
            "content": "",
            "reasoning_content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "上海"}
                    },
                    "type": "function",
                    "index": 0
                },
                {
                    "function": {
                        "name": "search_web",
                        "arguments": {"query": "最新AI新闻"}
                    },
                    "type": "function",
                    "index": 1
                }
            ]
        },
        {  
            # role和dataset配置中observation_tag对应,
            # 支持多个工具回复, 会自动合并为一个observation
            "role": "tool",
            "content": "{\"result\": \"28°C, 多云\"}",
            "tool_call_id": 0
        },
        {
            "role": "tool",
            "content": "{\"result\": \"2025年GPT-5发布在即...\"}",
            "tool_call_id": 1
        },
        {
            "role": "assistant",
            "content": "上海当前天气：28°C，多云。最新AI新闻：2025年GPT-5发布在即，引发行业热议。",
            "reasoning_content": "",
            "tool_calls": []
        }
    ]
}
```
