"""End-to-end token consistency for llmir_paged vs HuggingFace (network)."""

import pytest

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def opt125m_ids():
    pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "facebook/opt-125m"
    prompt = "The capital of France is"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=__import__("torch").float32, device_map="cpu"
    ).eval()
    inp = tok(prompt, return_tensors="pt")
    prompt_len = inp.input_ids.shape[1]
    with __import__("torch").no_grad():
        hf_out = model.generate(
            **inp,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    hf_gen = hf_out[0, prompt_len:].tolist()

    from llmir import LLMEngine, SamplingParams

    engine = LLMEngine.from_pretrained(model_id, backend="llmir_paged", dtype="float32")
    out = engine.generate(
        [prompt], SamplingParams(max_tokens=8, temperature=0.0), use_tqdm=False
    )
    llmir_gen = out[0].outputs[0].token_ids
    engine.shutdown()
    return hf_gen, llmir_gen


def test_llmir_paged_matches_hf_greedy_tokens(opt125m_ids):
    hf_gen, llmir_gen = opt125m_ids
    assert hf_gen == llmir_gen
