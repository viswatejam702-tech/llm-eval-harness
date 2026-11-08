# Aegis Harness — System Boundaries & Limitations

## 1. Probabilistic Nature of LLMs
Non-zero temperature generation can cause minor evaluation variance across runs. For definitive pre-deployment certification, Aegis recommends running evaluations at temperature $T = 0.0$ or averaging metrics across $N \ge 3$ repeated runs.

## 2. Dynamic Red-Teaming Coverage
While the 30 standardized adversarial attack vectors and 100 consistency clusters cover major vulnerability classes (OWASP Top 10 for LLMs), novel zero-day prompt injection primitives require continuous dataset updates.

## 3. Sandboxed Agent Tool Execution
Tool sandboxing evaluates the model's intent, tool selection, and argument sanitization. It does not replace live containerized staging integration tests in enterprise deployment pipelines.
