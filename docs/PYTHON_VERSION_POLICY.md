# Python Version Policy

## Supported versions

| Context | Versions |
|---------|----------|
| **Package (`pyproject.toml`)** | `>=3.9` |
| **CI test matrix** | 3.9, 3.10, 3.11, 3.12 |
| **Lint / walkthrough CI** | 3.11 |

## Rationale (Loop R16)

Python **3.8** was removed from the CI matrix because:

- Modern `torch` / `transformers` stacks used by optional `[full]` extras target 3.9+
- New harness code uses PEP 585 annotations (`list[str]`, `tuple[str, ...]`) with `from __future__ import annotations`
- A-class walkthrough and lint gates are validated on 3.11; the matrix guards compatibility across 3.9–3.12

## Local check

```bash
python3 --version   # expect >= 3.9
bash scripts/ci_lint_gate.sh
bash scripts/walkthrough_a_class.sh
```
