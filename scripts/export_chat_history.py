import json
import os
from datetime import datetime

source_path = r"C:\Users\JOSHUA\.gemini\antigravity\brain\a362d791-ba8f-443a-81ee-a052f6e15cff\.system_generated\logs\transcript_full.jsonl"
if not os.path.exists(source_path):
    source_path = r"C:\Users\JOSHUA\.gemini\antigravity\brain\a362d791-ba8f-443a-81ee-a052f6e15cff\.system_generated\logs\transcript.jsonl"

out_path = r"c:\Users\JOSHUA\Desktop\10am\FULL_CHAT_HISTORY.md"

messages = []

with open(source_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        
        step_type = entry.get("type", "")
        source = entry.get("source", "")
        content = entry.get("content", "")
        created_at = entry.get("created_at", "")
        
        # We want user inputs and assistant responses
        if step_type == "USER_INPUT" or source == "USER_EXPLICIT":
            if content and not content.startswith("<CONTEXT_SUMMARY>") and not content.startswith("<SYSTEM_MESSAGE>"):
                messages.append({
                    "role": "USER (Josh)",
                    "time": created_at,
                    "content": content
                })
        elif step_type == "PLANNER_RESPONSE" or source == "MODEL":
            if content:
                messages.append({
                    "role": "ASSISTANT (Antigravity)",
                    "time": created_at,
                    "content": content
                })

print(f"Total conversation messages extracted: {len(messages)}")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Complete Conversation History — NQ Sniper & BTC Algorithmic System\n\n")
    f.write(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Repository:** https://github.com/Josh-devv/ict-930-sniper-bot\n")
    f.write(f"**Total Messages:** {len(messages)}\n\n")
    f.write("---\n\n")
    
    for i, msg in enumerate(messages, 1):
        role_header = f"### [{i}] {msg['role']}  `{msg['time'][:19] if msg['time'] else ''}`\n\n"
        f.write(role_header)
        f.write(msg['content'].strip() + "\n\n")
        f.write("---\n\n")

print(f"Successfully saved to {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")
