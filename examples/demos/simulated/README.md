# Simulated demos (not end-to-end inference)

Scripts here exercise LLMIR Python APIs with **synthetic data** or **placeholder**
generation. They do **not** load a real transformer or produce meaningful text.

For real model inference use:

```python
from llmir import LLMEngine, SamplingParams

engine = LLMEngine.from_pretrained("facebook/opt-125m", backend="llmir_paged")
print(engine.generate("Hello", SamplingParams(max_tokens=8))[0].outputs[0].text)
```

Reproducible benchmarks live under [`scripts/`](../../../scripts/).
