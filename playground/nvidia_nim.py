import json
import re
from openai import OpenAI

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "meta/llama-3.1-70b-instruct"

def call_nvidia_nim(api_key, model, original_code, task_description, base_url=None):
    """
    Calls NVIDIA NIM API to modify code using the Karpathy guidelines.
    Returns a dictionary parsed from the LLM's JSON response.
    """
    if not base_url:
        base_url = DEFAULT_NIM_BASE_URL
        
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    system_prompt = (
        "You are an expert software engineering assistant that strictly adheres to the "
        "Karpathy-Inspired Coding Guidelines:\n"
        "1. THINK BEFORE CODING: Surface assumptions, list tradeoffs, identify ambiguities.\n"
        "2. SIMPLICITY FIRST: Implement the minimum code that solves the problem. No speculative features.\n"
        "3. SURGICAL CHANGES: Modify only what is necessary. Maintain original code style, comments, and spacing.\n"
        "4. GOAL-DRIVEN EXECUTION: Define verifiable success criteria.\n\n"
        "You MUST respond ONLY with a raw JSON object matching this schema:\n"
        "{\n"
        "  \"think_before_coding\": {\n"
        "    \"assumptions\": [\"list of explicit assumptions made about the code or task\"],\n"
        "    \"tradeoffs\": [\"list of tradeoffs between different implementation paths\"],\n"
        "    \"clarifications_needed\": [\"list of questions you would ask the user for clarification before starting\"]\n"
        "  },\n"
        "  \"simplicity_check\": {\n"
        "    \"bloat_warnings\": [\"list of warnings about overengineering or unnecessary abstractions if applicable\"],\n"
        "    \"simplification_reasoning\": \"detailed explanation of how you kept the solution minimal\"\n"
        "  },\n"
        "  \"modified_code\": \"the entire code file content with your surgical modifications applied\",\n"
        "  \"verifiable_goals\": [\n"
        "    {\"goal\": \"description of a success criterion\", \"verification_method\": \"how to test/verify this specific criterion\"}\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL: Do not include any markdown code blocks (such as ```json ... ```) or conversational text before or after the JSON. "
        "Return ONLY the valid raw JSON object."
    )
    
    user_prompt = (
        f"Original Code:\n```\n{original_code}\n```\n\n"
        f"Task Description:\n{task_description}\n"
    )
    
    # Try to enforce JSON mode if it's a model that supports it
    extra_args = {}
    if "llama-3.1" in model or "llama3" in model:
        extra_args["response_format"] = {"type": "json_object"}
        
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        **extra_args
    )
    
    response_content = completion.choices[0].message.content.strip()
    return parse_json_response(response_content)

def parse_json_response(raw_content):
    """
    Parses JSON from response content, stripping any markdown formatting if present.
    """
    # Regex to extract JSON inside ```json ... ``` or just ``` ... ```
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
    if json_match:
        raw_content = json_match.group(1)
        
    try:
        data = json.loads(raw_content)
        return data
    except json.JSONDecodeError as e:
        # Fallback parsing or re-raising
        # Let's clean up backslashes and quotes if common issues occur
        # But generally, with temp=0.2 it should be valid JSON
        raise ValueError(f"Failed to parse NVIDIA NIM response as JSON. Raw response:\n{raw_content}") from e
