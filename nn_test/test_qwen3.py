import json
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nn.models

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"

# System prompt configuration
SYSTEM_PROMPT_TEMPLATE = """你是一个智能助手，负责将用户输入转换为结构化的JSON动作指令。

## 支持的动作类型：

1. **replay** - 智能体直接回复用户，无需调用大模型
   - 参数: content (string) - 直接回复用户的内容

2. **get_llm_response** - 将用户原始输入转换为适合大模型理解的提示词
   - 参数: prompt (string) - 融合智能体兴趣（如：原神的甘雨）和用户输入（如：你喜欢哪个游戏里的角色），生成适合大模型理解的提示词。例如："你喜欢原神的甘雨，用户询问你喜欢什么游戏角色，生成回复"

3. **adjust_reward** - 调整奖励因子
   - 参数: content (string), factor (number) - 内容和调整因子(-1.0到1.0)

## 智能体上下文信息：
智能体感兴趣的内容: {high_reward_content}
(这些是智能体通过学习和交互积累的偏好记忆，代表智能体关注和感兴趣的话题领域)

## 输出格式：
请严格按照以下JSON格式输出（支持单个或多个动作）：

单个动作：
```json
{{
  "action": "动作类型",
  "parameters": {{
    "相关参数": "参数值"
  }}
}}
```

多个动作：
```json
{{
  "actions": [
    {{
      "action": "动作类型1",
      "parameters": {{
        "相关参数": "参数值"
      }}
    }},
    {{
      "action": "动作类型2", 
      "parameters": {{
        "相关参数": "参数值"
      }}
    }}
  ]
}}
```

用法说明：如果可以直接用 replay 回复用户，则优先生成 replay 动作；只有在需要大模型理解和生成时，才使用 get_llm_response。
请根据用户输入选择最合适的动作并生成对应的JSON指令。"""


def create_system_prompt(
    high_reward_content: str = "讨厌华为手机,因为爱国营销太傻逼了",
) -> str:
    """创建系统提示词"""
    return SYSTEM_PROMPT_TEMPLATE.format(high_reward_content=high_reward_content)


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """加载模型和分词器"""
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

    # Get device and move model to it
    device = nn.models.get_device()
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    return model, tokenizer, device


def prepare_messages(
    user_prompt: str, system_prompt: str = None
) -> List[Dict[str, str]]:
    """准备对话消息"""
    if system_prompt is None:
        system_prompt = create_system_prompt()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# Load model and tokenizer
model, tokenizer, device = load_model_and_tokenizer()

# Prepare the model input
user_prompt = "华为和小米的手机，选择一个"
messages = prepare_messages(user_prompt)


def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    device,
    max_new_tokens: int = 32768,
    enable_thinking: bool = False,
) -> Dict[str, str]:
    """生成模型响应"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    # Parse thinking content if enabled
    thinking_content = ""
    content = ""

    if enable_thinking:
        try:
            # Find </think> token (151668) - following Qwen's official approach
            index = len(output_ids) - output_ids[::-1].index(151668)
            # Use Qwen's official parsing method
            thinking_content = tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")
            content = tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
        except ValueError:
            # No thinking tokens found, treat entire output as content
            raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            content = raw_output.strip("\n")
    else:
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        content = raw_output.strip("\n")

    # Keep raw output for debugging
    raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {
        "thinking": thinking_content,
        "response": content,
        "raw_output": raw_output,  # Keep raw output for debugging
    }


def parse_json_response(response: str, thinking_content: str = "") -> Dict[str, Any]:
    """解析JSON响应，优先从思考过程中提取JSON"""

    def try_parse_json(text: str) -> Dict[str, Any]:
        """尝试解析JSON文本"""
        # Method 1: Look for ```json blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Method 2: Look for { } blocks
        start_idx = text.find("{")
        if start_idx != -1:
            # Find the matching closing brace
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            if brace_count == 0:
                json_str = text[start_idx:end_idx]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Method 3: Try parsing the entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        return None

    # Try parsing from thinking content first (often contains more structured reasoning)
    if thinking_content:
        result = try_parse_json(thinking_content)
        if result:
            return {
                "source": "thinking",
                "data": result,
                "thinking_content": thinking_content,
            }

    # Try parsing from response
    result = try_parse_json(response)
    if result:
        return {"source": "response", "data": result}

    # If all parsing fails, return error info
    return {
        "error": "Failed to parse JSON",
        "raw_response": response,
        "thinking_content": thinking_content,
        "source": "none",
    }


def display_results(result: Dict[str, str], parsed_json: Dict[str, Any]):
    """美化显示结果"""
    print("🤖 " + "=" * 60)
    print("🧠 THINKING PROCESS:")
    print("=" * 60)

    if result["thinking"]:
        # Split thinking into paragraphs for better readability
        thinking_lines = result["thinking"].split("\n")
        for line in thinking_lines:
            if line.strip():
                print(f"💭 {line.strip()}")
        print()
    else:
        print("💭 No thinking process available (thinking mode disabled)")
        print()

    print("📝 " + "=" * 60)
    print("💬 FINAL RESPONSE:")
    print("=" * 60)
    print(result["response"])
    print()

    print("🔍 " + "=" * 60)
    print("📊 PARSED JSON RESULT:")
    print("=" * 60)

    if "error" not in parsed_json:
        print(
            f"✅ JSON parsed successfully from: {parsed_json.get('source', 'unknown')}"
        )
        print("📋 Structured output:")
        print(
            json.dumps(
                parsed_json.get("data", parsed_json), indent=2, ensure_ascii=False
            )
        )

        # Analyze the action
        if "data" in parsed_json and "action" in parsed_json["data"]:
            action = parsed_json["data"]["action"]
            params = parsed_json["data"].get("parameters", {})
            print(f"\n🎯 Action detected: {action}")
            print(f"⚙️  Parameters: {json.dumps(params, ensure_ascii=False)}")
    else:
        print("❌ Failed to parse JSON")
        print(f"🔍 Attempted to parse from: {parsed_json.get('source', 'unknown')}")
        if parsed_json.get("thinking_content"):
            print("🧠 Thinking content was available but didn't contain valid JSON")
    print("=" * 60)


# Generate response
print("🚀 Generating response with thinking process enabled...")
print(f"📝 User prompt: {user_prompt}")
print("⏳ Processing...")
print()

result = generate_response(model, tokenizer, messages, device, enable_thinking=False)

# Parse JSON from both thinking and response
parsed_result = parse_json_response(result["response"], result["thinking"])

# Display results in a beautiful format
display_results(result, parsed_result)

# Additional debugging info if needed
if parsed_result.get("error"):
    print("\n🔧 DEBUG INFO:")
    print("-" * 40)
    print("Raw model output:")
    print(result.get("raw_output", "Not available"))
