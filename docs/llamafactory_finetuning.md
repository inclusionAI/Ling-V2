## LLaMA-Factory Use Documentation

This patch supports the OpenAI format and fc data with multiple tool calls

### Clone Repository

```bash
# Clone LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Checkout specific commit
git checkout d891bfedad9b3f3ca439af0180bdb6c0b1cfd4ab

# Install
pip install -e ".[torch,metrics]" --no-build-isolation
```

### Configuration Instructions

Fine-tuning yaml `template` filed set to `bailing_v2`

Add the following dataset configuration to `data/dataset_info.json`, modify the corresponding parameters according to the instructions, and do not modify the default values if they exist

Other `sharegpt` format data can be modified by changing `formatting` to `openai` to be compatible with Ling model training

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "ms_hub_url": "the name of the dataset repository on the Model Scope hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name and cloud_file_name)",
  "cloud_file_name": "the name of the dataset file in s3/gcs cloud storage. (if specified, ignore file_name)",
  "file_name": "the name of the dataset folder or dataset file in this directory. (required if above are not specified)",
  "formatting": "openai",
  "ranking": "whether the dataset is a preference dataset or not. (default: False)",
  "subset": "the name of the subset. (optional, default: None)",
  "split": "the name of dataset split to be used. (optional, default: train)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "num_samples": "the number of samples in the dataset to be used. (optional, default: None)",
  "columns": {
    "messages": "messages",
    "tools": "tools",
    "images": "the column name in the dataset containing the image inputs. (default: None)",
    "videos": "the column name in the dataset containing the videos inputs. (default: None)",
    "audios": "the column name in the dataset containing the audios inputs. (default: None)",
    "chosen": "the column name in the dataset containing the chosen answers. (default: None)",
    "rejected": "the column name in the dataset containing the rejected answers. (default: None)",
    "kto_tag": "the column name in the dataset containing the kto tags. (default: None)"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "observation_tag": "tool",  # Correspond to the tool response role in the data
    "function_tag": "function_call",  # Correspond to the tool calls role in the data
    "system_tag": "system"
  }
}
```


### SFT Demo

We use the [glaive_toolcall_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/glaive_toolcall_en_demo.json) dataset to demonstrate how to fine-tune the Ling model. You need to modify the `formatting` setting for `glaive_toolcall_en_demo` to `openai` in the [dataset_info.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/dataset_info.json) file.

We provide a demo configuration to start the SFT training for Ling:

`llamafactory-cli train examples/sft/llama_factory/ling_full_sft.yaml`

### Data Format

Single data example:

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

Data with additional explanation:

```python
{
    # The "tools" field supports either a string or a list after json dumps. When tools is not used, set it to empty
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
            "role": "system",  # if not use `system`, can set it to empty
            "content": "default system"
        },
        {
            "role": "user",
            "content": "查一下上海天气和最新AI新闻"
        },
        {  
            # Tool calls must be configured in the following format: role correspond to function_tag (or assistant_tag) in the dataset configuration
            # The content is empty, and tool_calls is a list where each item contains a `function` field with specific tool call information
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
            # Role correspond to the "observation_tag" in the dataset configuration
            # Support multiple tool responses, which will be automatically merged into one observation
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
