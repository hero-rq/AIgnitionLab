# llm_engine.py

import time as time_ops
import tiktoken as tk_encoding
from openai import OpenAI as LLMOpenAI
import openai as openai_core
import os as sys_ops
import anthropic as anthro_api
import json as json_mgr

# Tracking token usage in/out per model
TOKEN_LOG_IN = {}
TOKEN_LOG_OUT = {}

# Initialize a Tiktoken encoding object
encoding_base = tk_encoding.get_encoding("cl100k_base")

def estimate_current_cost():
    """
    Estimate the approximate cost of the current experiment based on 
    tokens used per model, referencing a fixed cost map.
    (Values are placeholders and may not reflect real-time pricing.)
    """
    cost_map_in = {
        "gpt-4o": 2.50 / 1_000_000,
        "gpt-4o-mini": 0.150 / 1_000_000,
        "o1-preview": 15.00 / 1_000_000,
        "o1-mini": 3.00 / 1_000_000,
        "claude-3-5-sonnet": 3.00 / 1_000_000,
        "deepseek-chat": 1.00 / 1_000_000,
        "o1": 15.00 / 1_000_000,
    }
    cost_map_out = {
        "gpt-4o": 10.00 / 1_000_000,
        "gpt-4o-mini": 0.6 / 1_000_000,
        "o1-preview": 60.00 / 1_000_000,
        "o1-mini": 12.00 / 1_000_000,
        "claude-3-5-sonnet": 12.00 / 1_000_000,
        "deepseek-chat": 5.00 / 1_000_000,
        "o1": 60.00 / 1_000_000,
    }
    total_cost = 0.0
    for model_name in TOKEN_LOG_IN:
        total_cost += cost_map_in.get(model_name, 0.0) * TOKEN_LOG_IN[model_name]
    for model_name in TOKEN_LOG_OUT:
        total_cost += cost_map_out.get(model_name, 0.0) * TOKEN_LOG_OUT[model_name]

    return total_cost

def ask_model_engine(
    model_str, 
    prompt, 
    system_prompt, 
    openai_api_key=None, 
    anthropic_api_key=None, 
    tries=5, 
    timeout=5.0, 
    temp=None, 
    print_cost=True, 
    version="1.5"
):
    """
    Query an LLM model with the given prompt and system prompt. 
    Supports multiple backends, including various OpenAI and Anthropic endpoints.
    Tracks token usage for approximate cost calculations.
    """
    # Load environment key if none was passed
    maybe_openai_key = sys_ops.getenv('OPENAI_API_KEY')
    if openai_api_key is None and maybe_openai_key is not None:
        openai_api_key = maybe_openai_key
    if openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in ask_model_engine function.")

    # Set environment variables if applicable
    if openai_api_key is not None:
        openai_core.api_key = openai_api_key
        sys_ops.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        sys_ops.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Attempt to get a response, with retries
    for _ in range(tries):
        try:
            answer = None

            # GPT-4o-mini
            if model_str in ["gpt-4o-mini", "gpt4omini", "gpt-4omini", "gpt4o-mini"]:
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                # If using older openai version:
                if version == "0.28":
                    if temp is None:
                        completion = openai_core.ChatCompletion.create(
                            model=model_str,
                            messages=messages
                        )
                    else:
                        completion = openai_core.ChatCompletion.create(
                            model=model_str,
                            messages=messages,
                            temperature=temp
                        )
                else:
                    # Use LLMOpenAI as in your snippet
                    client = LLMOpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=messages
                        )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18",
                            messages=messages,
                            temperature=temp
                        )
                answer = completion.choices[0].message.content

            # Claude-3.5-Sonnet (Anthropic)
            elif model_str == "claude-3.5-sonnet":
                client = anthro_api.Anthropic(api_key=sys_ops.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                # Pseudocode—actual usage might differ
                answer = json_mgr.loads(message.to_json())["content"][0]["text"]

            # GPT-4o (gpt4o, gpt-4o)
            elif model_str in ["gpt4o", "gpt-4o"]:
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                if version == "0.28":
                    if temp is None:
                        completion = openai_core.ChatCompletion.create(
                            model=model_str,
                            messages=messages
                        )
                    else:
                        completion = openai_core.ChatCompletion.create(
                            model=model_str,
                            messages=messages,
                            temperature=temp
                        )
                else:
                    client = LLMOpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06",
                            messages=messages
                        )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06",
                            messages=messages,
                            temperature=temp
                        )
                answer = completion.choices[0].message.content

            # DeepSeek
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client.")
                else:
                    deepseek_client = LLMOpenAI(
                        api_key=sys_ops.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages
                        )
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp
                        )
                answer = completion.choices[0].message.content

            # o1-mini
            elif model_str == "o1-mini":
                messages = [
                    {"role": "user", "content": system_prompt + prompt}
                ]
                if version == "0.28":
                    completion = openai_core.ChatCompletion.create(
                        model=model_str,
                        messages=messages
                    )
                else:
                    client = LLMOpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12",
                        messages=messages
                    )
                answer = completion.choices[0].message.content

            # o1
            elif model_str == "o1":
                messages = [
                    {"role": "user", "content": system_prompt + prompt}
                ]
                if version == "0.28":
                    completion = openai_core.ChatCompletion.create(
                        model="o1-2024-12-17",
                        messages=messages
                    )
                else:
                    client = LLMOpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17",
                        messages=messages
                    )
                answer = completion.choices[0].message.content

            # o1-preview
            elif model_str == "o1-preview":
                messages = [
                    {"role": "user", "content": system_prompt + prompt}
                ]
                if version == "0.28":
                    completion = openai_core.ChatCompletion.create(
                        model=model_str,
                        messages=messages
                    )
                else:
                    client = LLMOpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview",
                        messages=messages
                    )
                answer = completion.choices[0].message.content

            # After we get the answer, track token usage
            if answer is not None:
                # Choose an encoding for cost approximation
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
                    chosen_enc = tk_encoding.encoding_for_model("gpt-4o")
                elif model_str in ["deepseek-chat"]:
                    chosen_enc = tk_encoding.encoding_for_model("cl100k_base")
                else:
                    chosen_enc = tk_encoding.encoding_for_model(model_str)

                # Init counters if not present
                if model_str not in TOKEN_LOG_IN:
                    TOKEN_LOG_IN[model_str] = 0
                    TOKEN_LOG_OUT[model_str] = 0

                # Count tokens in system+prompt vs tokens in response
                TOKEN_LOG_IN[model_str] += len(chosen_enc.encode(system_prompt + prompt))
                TOKEN_LOG_OUT[model_str] += len(chosen_enc.encode(answer))

                # Optionally print approximate cost
                if print_cost:
                    print(f"Current experiment cost = ${estimate_current_cost():.4f}, "
                          f"**Approx. values, may not reflect true cost**")

                return answer

        except Exception as exc:
            print("Inference Exception:", exc)
            time_ops.sleep(timeout)
            continue

    # If all attempts fail:
    raise Exception("Max retries exceeded: request timed out or failed.")
