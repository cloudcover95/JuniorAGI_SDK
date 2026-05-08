# /Users/nico/Documents/JuniorCloud/JuniorAGI_SDK/src/sandbox/llm_bridge.py

import mlx.core as mx
from mlx_lm import load, generate
import json

class AdaptiveLLMSandbox:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model, self.tokenizer = None, None
        self._load_model()

    def _load_model(self):
        try:
            self.model, self.tokenizer = load(self.model_path)
        except Exception as e:
            print(f"Node Failure: LLM Matrix allocation failed. Error: {e}")

    def generate_ui_payload(self, user_query: str) -> dict:
        """
        Forces LLM to output a Generative UI schema for frontend mounting.
        If 'calendar' is detected, returns the Rigid Orchestrator schema.
        """
        if "calendar" in user_query.lower():
            return {
                "type": "ui_component",
                "component": "RigidCalendarGrid",
                "props": {
                    "theme": "tradingview_dark",
                    "data_source": "parquet://ts_lakes/calendar_meta",
                    "sync_rate": "20Hz"
                }
            }

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the JuniorCloud LLC UI Orchestrator. Output ONLY valid JSON.
If the user requests a specific tool, output a UI mounting schema.
Schema Format: {{"type": "ui_component", "component": "Name", "props": {{}}}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=150, verbose=False)
        try:
            return json.loads(response.strip())
        except:
            return {"type": "text", "content": response}
