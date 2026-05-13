import gradio as gr, sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.injection_pipeline import HybridLLMInjector
from kernel.agi_kernel import JuniorAGI

inj = HybridLLMInjector(JuniorAGI())

def process(f):
    if not f: return "No file", "Waiting..."
    t0 = time.perf_counter()
    res = inj.inject_safetensors(f.name)
    out = f"HMR Complete ({time.perf_counter()-t0:.2f}s)\nGPU Layers: {res.get('gpu_layers')}\nCPU Layers: {res.get('cpu_layers')}"
    return "Success", out

with gr.Blocks(title="JuniorAGI HMR") as app:
    gr.Markdown("# Sovereign Checkpoint Injection (HMR)")
    with gr.Row():
        f_in = gr.File(label=".safetensors")
        btn = gr.Button("Inject")
    with gr.Row():
        stat = gr.Textbox(label="Status")
        log = gr.TextArea(label="Console")
    btn.click(fn=process, inputs=f_in, outputs=[stat, log])
if __name__ == "__main__": app.launch(server_name="0.0.0.0", server_port=8080)
