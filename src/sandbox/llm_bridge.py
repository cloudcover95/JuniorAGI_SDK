# /Users/nico/Documents/JuniorCloud/JuniorAGI_SDK/src/sandbox/llm_bridge.py

import mlx.core as mx
from mlx_lm import load, generate

class AdaptiveLLMSandbox:
    """
    The Local Open-Source LLM Integration Node.
    Acts as the semantic interface for the underlying mathematical engine.
    Adapts to the user's system constraints via Apple Silicon MLX.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model, self.tokenizer = None, None
        self._load_model()

    def _load_model(self):
        try:
            # Loads quantized model natively onto M4 unified memory
            self.model, self.tokenizer = load(self.model_path)
        except Exception as e:
            print(f"Node Failure: LLM Matrix allocation failed. Ensure mlx-lm is installed and model path is valid. Error: {e}")

    def generate_insight(self, topological_context: str, user_query: str) -> str:
        if not self.model:
            return "Sandbox Offline: Model not loaded."

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the JuniorCloud LLC Strategic Interface. 
Translate the deterministic hardware state into actionable engineering metrics.
System Context: {topological_context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        return response
