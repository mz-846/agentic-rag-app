"""Fake OpenAI-compatible chat endpoint used by the offline tests (stands in for Groq).

Behaviour: if the request offers tools and no tool result is in the conversation yet, it
asks to call the first tool (so tool plumbing gets exercised); otherwise it returns a
canned final answer. Every request is appended to a JSONL log for assertions.
"""
import json
import sys
import time

from flask import Flask, jsonify, request

app = Flask(__name__)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
LOG = sys.argv[2] if len(sys.argv) > 2 else "fake_llm_requests.jsonl"


@app.post("/v1/chat/completions")
def chat():
    body = request.get_json()
    with open(LOG, "a") as f:
        f.write(json.dumps(body) + "\n")
    tools = body.get("tools")
    if tools and not any(m.get("role") == "tool" for m in body["messages"]):
        fn = tools[0]["function"]
        args = {k: "Sporo Health patient chart review" for k in fn.get("parameters", {}).get("properties", {})}
        msg = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": fn["name"], "arguments": json.dumps(args)}}]}
        finish = "tool_calls"
    else:
        msg = {"role": "assistant",
               "content": "FAKE ANSWER: Sporo Health uses AI agents to summarise patient charts [doc.pdf p.3]."}
        finish = "stop"
    return jsonify({"id": "fake", "object": "chat.completion", "created": int(time.time()),
                    "model": body["model"], "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})


if __name__ == "__main__":
    app.run(port=PORT)
