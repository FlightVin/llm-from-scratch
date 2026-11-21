"""Focused, CPU-only checks for the part notebooks.

The notebooks are the source of truth for the implementation. These tests read
the notebook JSON and execute selected code cells with a stub `mugrade` module,
so they need no GPU, no course datasets, no checkpoints, and no Mugrade account.

Run from the repository root:

    pytest
"""

import ast
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("part[0-6]_*/*.ipynb"))

# Stub for the Mugrade grading client: the decorators return the function as-is.
MUGRADE_STUB = types.SimpleNamespace(
    submit_tests=lambda function: function,
    local_tests=lambda function: function,
)


def _mugrade_name(name):
    return lambda function: function


def _code_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = cell["source"]
            yield "".join(source) if isinstance(source, list) else source


def _notebook(prefix):
    matches = sorted(ROOT.glob(f"{prefix}*/*.ipynb"))
    assert len(matches) == 1, f"expected one notebook for {prefix!r}, found {matches}"
    return matches[0]


def _cell_defining(path, pattern):
    for source in _code_cells(path):
        if re.search(pattern, source, re.MULTILINE):
            return source
    raise LookupError(f"no code cell in {path.name} matches {pattern!r}")


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------

def test_one_notebook_per_part():
    assert [p.parent.name.split("_")[0] for p in NOTEBOOKS] == [f"part{i}" for i in range(7)]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_notebook_is_nbformat_v4(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all({"cell_type", "source"} <= set(cell) for cell in notebook["cells"])


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_no_hardcoded_mugrade_key(path):
    literal_key = re.compile(r"""MUGRADE_KEY["']\s*\]\s*=\s*["'][^"']+["']""")
    for source in _code_cells(path):
        assert not literal_key.search(source), "read MUGRADE_KEY from the environment"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.parent.name)
def test_opened_config_files_are_written_or_downloaded(path):
    """Every literal config path that a cell opens must be produced earlier."""
    sources = list(_code_cells(path))
    provided = set()
    for source in sources:
        provided.update(re.findall(r"^%%writefile\s+(\S+)", source, re.MULTILINE))
        if "hf_hub_download" in source:
            provided.update(re.findall(r"""["']([\w.\-]+\.json)["']""", source))
    for source in sources:
        for name in re.findall(r"""open\(\s*["'](config[\w.\-]*\.json)["']""", source):
            assert name in provided, f"{name} is opened but never written or downloaded"


# ---------------------------------------------------------------------------
# LLM buffers: every buffer that a method reads must be built in __init__
# ---------------------------------------------------------------------------

def _buffer_keys(function, store):
    keys = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "buffers"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.ctx, ast.Store) == store
        ):
            keys.add(node.slice.value)
    return keys


def _llm_classes():
    for path in NOTEBOOKS:
        for source in _code_cells(path):
            if "class LLM" not in source:
                continue
            for node in ast.parse(source).body:
                if isinstance(node, ast.ClassDef) and node.name == "LLM":
                    yield pytest.param(node, id=f"{path.parent.name}:L{node.lineno}")


@pytest.mark.parametrize("llm_class", list(_llm_classes()))
def test_llm_reads_only_buffers_built_in_init(llm_class):
    methods = {f.name: f for f in llm_class.body if isinstance(f, ast.FunctionDef)}
    built = _buffer_keys(methods["__init__"], store=True)
    read = set()
    for name, function in methods.items():
        if name != "__init__":
            read |= _buffer_keys(function, store=False)
    assert read <= built, f"buffers read but never built: {sorted(read - built)}"


# ---------------------------------------------------------------------------
# Part 1: BPE tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bpe_class():
    source = _cell_defining(_notebook("part1_"), r"^class BPE:")
    namespace = {"mugrade": MUGRADE_STUB}
    tqdm_stub, tqdm_auto_stub = types.ModuleType("tqdm"), types.ModuleType("tqdm.auto")
    tqdm_auto_stub.tqdm = lambda iterable, **kwargs: iterable
    tqdm_stub.auto = tqdm_auto_stub
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "tqdm", tqdm_stub)
        patch.setitem(sys.modules, "tqdm.auto", tqdm_auto_stub)
        exec(compile(source, "part1:bpe", "exec"), namespace)
    return namespace["BPE"]


def test_bpe_round_trip_keeps_special_tokens(bpe_class, tmp_path):
    samples = [
        "<doc>Hey hey hey hey there lads lads and ladies",
        "<doc> Hey hey hey hey there lads lads and ladies",
        "Hey hey<doc> hey hey there lads lads and ladies",
        "Hey hey hey hey there lads lads and ladies <doc>",
    ]
    bpe = bpe_class()
    bpe.special_tokens[260] = "<doc>"
    bpe.train(samples[0], 260)
    assert len(bpe.merges) == 4 and bpe.vocab[260] == "<doc>"

    for text in samples:
        tokens = bpe.encode(text)
        assert tokens.count(260) == 1, "the special token must encode to its own ID"
        assert bpe.decode(tokens) == text

    path = tmp_path / "tiny.bpe"
    bpe.save(path)
    loaded = bpe_class(path)
    assert loaded.special_tokens == {260: "<doc>"}
    assert all(loaded.encode(text) == bpe.encode(text) for text in samples)


# ---------------------------------------------------------------------------
# Part 5: calculator tool used during generation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tool_eval():
    source = _cell_defining(_notebook("part5_"), r"^def tool_eval\(")
    namespace = {"mugrade": MUGRADE_STUB, "re": re, "json": json}
    exec(compile(source, "part5:tool_eval", "exec"), namespace)
    return namespace["tool_eval"]


@pytest.mark.parametrize(
    "expression, expected",
    [
        # Cases from part5_rl_tests.py (test_tool_eval and submit_tool_eval).
        ("2 + 3 * 4", 14),
        ("10 / 3", 3.33),
        ("-9 / 2", -4.5),
        ("1 / 0", ""),
        ("not valid arithmetic", ""),
        ("8 / 2", 4),
        ("(7 - 11) ** 2", 16),
        ("2 / 7", 0.29),
        ("3 +", ""),
        ("15 / 3", 5),
        # GSM8K-style calls with surrounding whitespace.
        (" 48/2 ", 24),
    ],
)
def test_tool_eval_matches_course_contract(tool_eval, expression, expected):
    result = tool_eval(expression)
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "open('README.md').read()",
        "(lambda: 1)()",
        "[1, 2, 3]",
        "x + 1",
        "9 ** 9 ** 9",
    ],
)
def test_tool_eval_rejects_non_arithmetic(tool_eval, expression):
    assert tool_eval(expression) == ""


# ---------------------------------------------------------------------------
# Part 6: autodiff engine, run on NumPy in place of CuPy
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def autodiff():
    numpy = pytest.importorskip("numpy")
    source = _cell_defining(_notebook("part6_"), r"^class Tensor:")
    namespace = {"mugrade": MUGRADE_STUB, "_mugrade_name": _mugrade_name}
    with pytest.MonkeyPatch.context() as patch:
        # The cell runs `import torch; del torch` and `import cupy as np`.
        patch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        patch.setitem(sys.modules, "cupy", numpy)
        exec(compile(source, "part6:autodiff", "exec"), namespace)
    assert namespace["np"] is numpy
    return namespace


def _assert_gradients(autodiff, loss_fn, arrays, eps, rtol, atol):
    """Compare Tensor.backward() with central finite differences."""
    numpy = autodiff["np"]
    Tensor = autodiff["Tensor"]

    leaves = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    loss_fn(*leaves).backward()

    def loss_value(values):
        return float(loss_fn(*[Tensor(v.copy(), requires_grad=True) for v in values]).value)

    for i, leaf in enumerate(leaves):
        numeric = numpy.zeros_like(arrays[i])
        for index in numpy.ndindex(arrays[i].shape):
            original = arrays[i][index]
            arrays[i][index] = original + eps
            plus = loss_value(arrays)
            arrays[i][index] = original - eps
            minus = loss_value(arrays)
            arrays[i][index] = original
            numeric[index] = (plus - minus) / (2 * eps)
        numpy.testing.assert_allclose(leaf.grad, numeric, rtol=rtol, atol=atol)


def test_backward_accepts_plain_array_targets(autodiff):
    numpy = autodiff["np"]
    logits_value = numpy.array([[2.0, -1.0, 0.5], [0.0, 1.0, -2.0]])
    targets = numpy.array([0, 2])  # A plain array, not a Tensor.

    logits = autodiff["Tensor"](logits_value.copy(), requires_grad=True)
    autodiff["cross_entropy_loss"](logits, targets).backward()

    shifted = numpy.exp(logits_value - logits_value.max(axis=-1, keepdims=True))
    expected = shifted / shifted.sum(axis=-1, keepdims=True)
    expected[numpy.arange(2), targets] -= 1.0
    numpy.testing.assert_allclose(logits.grad, expected / 2, rtol=1e-12, atol=1e-12)


def test_gradients_mlp_path(autodiff):
    numpy = autodiff["np"]
    rng = numpy.random.default_rng(0)
    targets = rng.integers(0, 5, size=(2, 3))

    def loss_fn(x, w):
        hidden = autodiff["silu"](autodiff["rms_norm"](x) @ w)
        return autodiff["cross_entropy_loss"](hidden, targets)

    # rms_norm computes in float32, so use a larger step and looser tolerance.
    arrays = [rng.standard_normal((2, 3, 4)), rng.standard_normal((4, 5))]
    _assert_gradients(autodiff, loss_fn, arrays, eps=1e-3, rtol=1e-3, atol=1e-4)


def test_gradients_attention_path(autodiff):
    numpy = autodiff["np"]
    Tensor = autodiff["Tensor"]
    rng = numpy.random.default_rng(1)
    mask = Tensor(numpy.triu(numpy.full((3, 3), -numpy.inf), k=1))
    targets = rng.integers(0, 4, size=(1, 2, 3))

    def loss_fn(q, k, v):
        out = autodiff["self_attn"](q, k, v, mask)
        return autodiff["cross_entropy_loss"](out, targets)

    arrays = [rng.standard_normal((1, 2, 3, 4)) for _ in range(3)]
    _assert_gradients(autodiff, loss_fn, arrays, eps=1e-6, rtol=1e-5, atol=1e-8)


def test_gradients_embedding_with_repeated_indices(autodiff):
    numpy = autodiff["np"]
    rng = numpy.random.default_rng(2)
    indices = numpy.array([[1, 3, 1]])  # Index 1 repeats; its gradients must add.
    targets = rng.integers(0, 5, size=(1, 3))

    def loss_fn(table, w):
        hidden = table[indices].transpose(1, 2).reshape(1, 4, 3).transpose(1, 2)
        return autodiff["cross_entropy_loss"](hidden @ w, targets)

    arrays = [rng.standard_normal((5, 4)), rng.standard_normal((4, 5))]
    _assert_gradients(autodiff, loss_fn, arrays, eps=1e-6, rtol=1e-5, atol=1e-8)
