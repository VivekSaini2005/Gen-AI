from openai import OpenAI
import json
import sys

client = OpenAI(
    api_key="AIzaSyBA9K_HbU2JV1SLio9Mr-WKcJ3WRs6bUnc",  
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are an AI assistant.

CRITICAL RULES:
- You MUST respond with VALID JSON TEXT only
- Do NOT return null
- Do NOT return empty content
- Do NOT use markdown
- Do NOT return an array
- Do NOT explain outside JSON

JSON FORMAT (ALWAYS):
{ "step": "<STEP_NAME>", "content": "<string>" }

Allowed steps:
- START
- PLAN
- OUTPUT
"""

def safe_json_parse(text):
    try:
        return json.loads(text)
    except Exception:
        print("❌ Invalid JSON from model:")
        print(text)
        return None

def run_step(step_name, user_query, history):
    history.append({
        "role": "user",
        "content": f"STEP: {step_name}\nUSER_QUERY: {user_query}"
    })

    response = client.chat.completions.create(
        model="gemini-3-flash-preview",
        messages=history
    )

    raw = response.choices[0].message.content

    if not raw:
        raise RuntimeError("Model returned empty content")

    parsed = safe_json_parse(raw)
    if not parsed:
        raise RuntimeError("Model did not return valid JSON")

    history.append({
        "role": "assistant",
        "content": raw
    })

    return parsed

def main():
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    user_query = input("👉🏻 ").strip()
    if not user_query:
        print("❌ Empty input")
        sys.exit(1)

    # STEP 1: START
    start = run_step("START", user_query, message_history)
    print("🔥", start["content"])

    # STEP 2: PLAN
    plan = run_step("PLAN", user_query, message_history)
    print("🧠", plan["content"])

    # STEP 3: OUTPUT
    output = run_step("OUTPUT", user_query, message_history)
    print("🤖", output["content"])

if __name__ == "__main__":
    main()
