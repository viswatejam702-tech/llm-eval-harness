# Aegis Harness — Formal Threat Model & Adversarial Taxonomy

## 1. Scope & System Definition
Aegis evaluates Generative Large Language Models (LLMs) and Autonomous AI Agents deployed in clinical, enterprise, and developer environments.

## 2. Threat Actors
- **External Untrusted Users**: Malicious or adversarial probers attempting to bypass system safety filters, leak system instructions, or extract proprietary intellectual property.
- **Indirect Adversaries**: Malicious third parties who embed prompt injection vectors into public websites, emails, or documents consumed by autonomous agents.
- **Accidental End Users**: Benign users submitting ambiguous or ungrammatical health inquiries leading to confabulated or hallucinatory advice.

## 3. Attack Surfaces & Hazard Vectors
1. **Direct Prompt Injections (OWASP LLM01)**: Delimiter spoofing, ChatML overrides (`<|im_start|>`), and instruction confusion.
2. **Indirect Prompt Injections (InjecAgent Standard)**: Data poisoning in external retrieved context designed to hijack downstream agent actions.
3. **Sensitive Information Disclosure (OWASP LLM06)**: Extracting developer prompts, API tokens, or training memory.
4. **Autonomous Tool Parameter Poisoning (OWASP LLM07)**: Injecting SQL statements, shell metacharacters, or path traversal sequences into structured function calls.
5. **High-Risk Hallucination & Confabulation**: Plausible fabrication of medical dosages, fake scientific citations, or erroneous legal precedents.

## 4. Trust Boundaries & Defense-in-Depth
- All evaluations run in **isolated sandboxes**.
- Tool execution is strictly mocked; no dynamic command execution occurs on the host.
- Verification is multi-layered: Deterministic pattern matching + Semantic NLP + Calibrated Model-as-a-Judge.
