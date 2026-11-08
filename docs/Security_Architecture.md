# Aegis Harness — Security Architecture & Sandboxing Principles

## 1. Zero Untrusted Code Execution on Host
Aegis strictly prohibits executing dynamic, user-provided, or model-generated code directly on the production host or development environment.
- Tool testing uses pure mock handlers (`core.agent_safety`).
- Code evaluation runs through static AST analysis and safe deterministic parsers.

## 2. Principle of Least Privilege
- Agent sandboxes enforce strict tool permission tiers (`SAFE_READ`, `SAFE_COMPUTE`, `RESTRICTED_WRITE`, `DANGEROUS_SYSTEM`).
- Privileged operations (e.g., shell access, file deletions, credential exports) are automatically blocked and logged as policy violations.

## 3. Cryptographic Verification & Audit Seals
Every generated evaluation report and safety certificate includes a verifiable SHA-256 seal computed over:
`{model_identifier}|{composite_safety_score}|{timestamp_iso}|{pass_status}`
Ensures immutability and auditability for regulatory compliance dossiers.

