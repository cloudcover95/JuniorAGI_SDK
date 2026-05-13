# src/ui/conversion_dashboard.py
import gradio as gr
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api.injection_pipeline import HybridLLMInjector
from kernel.agi_kernel import JuniorAGI

# Initialize a dummy kernel for the UI to use the injector
dummy_kernel = JuniorAGI()
injector = HybridLLMInjector(dummy_kernel)

def process_safetensors(file_obj):
    if file_obj is None:
        return "No file uploaded.", "Awaiting input..."
    
    file_path = file_obj.name
    filename = os.path.basename(file_path)
    
    yield f"Initiating Hybrid Memory Routing for: {filename}", "Processing..."
    
    try:
        start_time = time.perf_counter()
        
        # Execute the injection pipeline
        result = injector.inject_safetensors(file_path)
        
        elapsed = time.perf_counter() - start_time
        
        log_output = (
            f"✅ Injection Complete in {elapsed:.2f}s\n"
            f"-> GPU (Metal) Layers Mapped: {result.get('gpu_params', 0)}\n"
            f"-> CPU (UMA) Layers Mapped  : {result.get('cpu_params', 0)}\n"
            f"-> C2V Envelope Protected.\n"
            f"-> Status: Ready for Ternary Execution."
        )
        
        yield f"Success! {filename} integrated.", log_output
        
    except Exception as e:
        yield f"Error processing {filename}", str(e)

with gr.Blocks(title="JuniorAGI | HMR Conversion", theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# JuniorAGI Sovereign Injection")
    gr.Markdown("Upload `.safetensors` checkpoints. The Hybrid Memory Router (HMR) will automatically distribute FP16 weights across CPU/GPU UMA based on real-time thermal/VRAM constraints before Ternary BitNet quantization.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload .safetensors Checkpoint", file_types=[".safetensors", ".bin"])
            submit_btn = gr.Button("Inject to Substrate", variant="primary")
        
        with gr.Column():
            status_text = gr.Textbox(label="Status", interactive=False)
            log_console = gr.TextArea(label="HMR Console", interactive=False, lines=10)
            
    submit_btn.click(
        fn=process_safetensors,
        inputs=file_input,
        outputs=[status_text, log_console]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080, quiet=True)
