
JuniorAGI // System Architecture & Mathematics
I. The Delta Rule & Information Density
Efficiency is calculated via Joules Per Inference (JPI). By mapping market sentiment to topological voids using .ply and .las formats, the system achieves a 20x JPI improvement over stochastic token prediction. High-Frequency Financial Modeling (HFFM) relies on the density of the state mesh, not the breadth of an LLM context window.

II. Adaptive Tensor Modulation
The reflection engine defaults to explicit Apple Silicon GPU/CPU streams via mlx.core.

Primary Path: mx.linalg.svd(A, stream=mx.cpu)

Cross-Platform Fallback: Ternary routing to numpy.linalg.svd for pure RAM/CPU hashing, ensuring the engine remains functional across Windows/Linux local nodes without architectural rewrite.

III. WebSockets & Generative UI
The UI is not statically bound. The LLM Sandbox acts as a schema-generator. When temporal coordination is required, the sandbox mounts the RigidCalendarGrid component dynamically over the WebSocket stream, locking to the 20Hz telemetry refresh rate.
