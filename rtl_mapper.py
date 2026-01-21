"""
rtl_mapper.py
==============

Production-quality CLI that maps a PyTorch model into a predefined RTL component library
and exports everything needed for RTL simulation:
- `netlist.json` (ordered RTL graph)
- quantized parameter `.mem` files (int16, hex lines)
- `mapping_report.txt` (human-readable summary/warnings)
- optional `rtl_wrapper.sv` skeleton instantiating RTL blocks in order

This repo's reference model/quantization matches `QuantizedMNISTNet (1).py`:
- SCALE_FACTOR default: 256
- weights/bias quantized to int16 with clamp to [-32768, 32767]
- quantize_tensor(x) = clamp(round(x * SCALE_FACTOR))/SCALE_FACTOR

### Quick start examples

- Map a Python-defined model class (state_dict `.pth` optional):

```bash
python rtl_mapper.py --model-module "QuantizedMNISTNet (1).py" --model-class SmallMNISTNet --model-path mnist_model.pth --out-dir ./SIM
```

- Map a TorchScript model (preferred for portability):

```bash
python rtl_mapper.py --torchscript ./model.ts --out-dir ./SIM
```

- Override example input shape (default MNIST `[1,1,28,28]`):

```bash
python rtl_mapper.py --model-module my_models.py --model-class MyNet --input-shape "1,3,32,32" --out-dir ./SIM
```

- Run a self-check (trains a tiny MNIST FC net, exports, re-loads exported params into a
Python fixed-point RTL-sim, and prints quick accuracy on 1000 test samples):

```bash
python rtl_mapper.py --self-test --out-dir ./SIM_SELFTEST
```
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("rtl_mapper")


# -----------------------------
# RTL library registry
# -----------------------------


@dataclass(frozen=True)
class RtlComponentSpec:
    """Specification of a supported RTL component."""

    name: str
    supported_ops: Tuple[str, ...]
    expected_params: Tuple[str, ...] = ()
    notes: str = ""


@dataclass
class RtlNode:
    """A node in the ordered RTL netlist graph."""

    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    in_shape: List[int]
    out_shape: List[int]
    params: Dict[str, Any] = field(default_factory=dict)


RTL_LIBRARY: Dict[str, RtlComponentSpec] = {
    "flatten": RtlComponentSpec(
        name="RTL_FLATTEN",
        supported_ops=("flatten",),
        expected_params=(),
        notes="Flattens NCHW to (N, 784) for MNIST-style input.",
    ),
    "linear": RtlComponentSpec(
        name="RTL_FC",
        supported_ops=("linear",),
        expected_params=("weight_mem", "bias_mem", "in_features", "out_features", "scale"),
        notes="Fully-connected GEMM + bias (fixed-point int16 params).",
    ),
    "relu": RtlComponentSpec(
        name="RTL_RELU",
        supported_ops=("relu",),
        expected_params=(),
        notes="ReLU activation (clamps negative to 0).",
    ),
}


# -----------------------------
# Quantization / .mem export
# -----------------------------


def _clamp_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767).astype(np.int16)


def float_to_int16(arr: np.ndarray, scale: int) -> np.ndarray:
    """Quantize float array to int16 using round(x * scale)."""
    return _clamp_int16(np.round(arr.astype(np.float32) * float(scale)))


def int16_to_float(arr: np.ndarray, scale: int) -> np.ndarray:
    return arr.astype(np.float32) / float(scale)


def quantize_tensor(t: torch.Tensor, scale: int) -> torch.Tensor:
    """Project tensor to fixed-point grid with given scale (and clamp to int16 range)."""
    return torch.clamp((t * scale).round(), -32768, 32767) / float(scale)


def write_mem_hex_int16(path: Path, arr_i16: np.ndarray) -> None:
    """Write int16 array to `.mem` as one 16-bit hex value per line (2's complement)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = arr_i16.reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFFFF:04x}\n")


def parse_input_shape(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --input-shape. Expected e.g. '1,1,28,28'.")
    shape = tuple(int(p) for p in parts)
    if any(d <= 0 for d in shape):
        raise ValueError(f"Invalid --input-shape {shape}. All dims must be positive.")
    return shape


# -----------------------------
# Model loading
# -----------------------------


def load_model_from_module(module_path: Path, class_name: str) -> nn.Module:
    """Load a Python class (nn.Module) from a filesystem path."""
    module_path = module_path.resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Model module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {module_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    if not hasattr(mod, class_name):
        raise AttributeError(f"Module '{module_path}' does not define class '{class_name}'.")
    cls = getattr(mod, class_name)
    model = cls()
    if not isinstance(model, nn.Module):
        raise TypeError(f"{class_name} is not a torch.nn.Module (got {type(model)}).")
    return model


def maybe_load_state_dict(model: nn.Module, model_path: Optional[Path]) -> None:
    """Load a saved state_dict if provided."""
    if model_path is None:
        return
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"--model-path not found: {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    model.load_state_dict(sd)


def load_torchscript(ts_path: Path) -> torch.jit.RecursiveScriptModule:
    ts_path = ts_path.resolve()
    if not ts_path.exists():
        raise FileNotFoundError(f"--torchscript not found: {ts_path}")
    return torch.jit.load(str(ts_path), map_location="cpu").eval()


# -----------------------------
# Op collection + shape inference
# -----------------------------


@dataclass
class CollectedOp:
    """A normalized op record, in execution order."""

    name: str
    op_kind: str  # 'flatten' | 'linear' | 'relu' | 'unsupported'
    module_qualname: Optional[str] = None  # for eager models (e.g. 'fc1')
    extra: Dict[str, Any] = field(default_factory=dict)
    in_shape: Optional[Tuple[int, ...]] = None
    out_shape: Optional[Tuple[int, ...]] = None


def _shape_to_list(shape: Optional[Tuple[int, ...]]) -> List[int]:
    return list(shape) if shape is not None else []


def _normalize_activation_name(name: str) -> str:
    # keep name stable and filesystem friendly for output files
    return "".join(c if (c.isalnum() or c in "_-.") else "_" for c in name)


def collect_ops_eager_fx(model: nn.Module, example_input: torch.Tensor) -> List[CollectedOp]:
    """
    Collect ops using torch.fx for eager models, then attach shapes using a single forward pass
    with FX shape propagation.
    """
    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception as e:
        raise RuntimeError(f"torch.fx.symbolic_trace failed. Try TorchScript input mode. Error: {e}") from e

    # First, collect module calls in topological order (the FX graph order).
    collected: List[CollectedOp] = []
    for n in gm.graph.nodes:
        if n.op == "call_module":
            submod = gm.get_submodule(str(n.target))
            if isinstance(submod, nn.Flatten):
                collected.append(CollectedOp(name=str(n.target), op_kind="flatten", module_qualname=str(n.target)))
            elif isinstance(submod, nn.Linear):
                collected.append(CollectedOp(name=str(n.target), op_kind="linear", module_qualname=str(n.target)))
            elif isinstance(submod, nn.ReLU):
                collected.append(CollectedOp(name=str(n.target), op_kind="relu", module_qualname=str(n.target)))
            elif isinstance(submod, (nn.Conv2d, nn.MaxPool2d, nn.Softmax)):
                collected.append(
                    CollectedOp(
                        name=str(n.target),
                        op_kind="unsupported",
                        module_qualname=str(n.target),
                        extra={"unsupported_module": type(submod).__name__},
                    )
                )
            else:
                # We don't fail immediately: some models contain Dropout/Identity/etc in eval, but they still
                # exist as ops. Treat them as unsupported to satisfy "stop with clear error" later.
                collected.append(
                    CollectedOp(
                        name=str(n.target),
                        op_kind="unsupported",
                        module_qualname=str(n.target),
                        extra={"unsupported_module": type(submod).__name__},
                    )
                )
        elif n.op == "call_function":
            # Typical patterns:
            # - torch.flatten / torch.Tensor.flatten / operator.getitem
            # - torch.nn.functional.relu
            tgt = n.target
            tgt_name = getattr(tgt, "__name__", str(tgt))
            if "flatten" in tgt_name:
                collected.append(CollectedOp(name=f"fx_{n.name}", op_kind="flatten"))
            elif "relu" in tgt_name:
                collected.append(CollectedOp(name=f"fx_{n.name}", op_kind="relu"))
            elif "linear" in tgt_name:
                collected.append(CollectedOp(name=f"fx_{n.name}", op_kind="linear"))
            else:
                collected.append(CollectedOp(name=f"fx_{n.name}", op_kind="unsupported", extra={"call_function": tgt_name}))
        elif n.op == "call_method":
            if str(n.target) == "flatten":
                collected.append(CollectedOp(name=f"fx_{n.name}", op_kind="flatten"))
            else:
                collected.append(CollectedOp(name=f"fx_{n.name}", op_kind="unsupported", extra={"call_method": str(n.target)}))

    # Next, attach shapes via FX ShapeProp so functional ops (torch.flatten/F.relu/etc) also get shapes.
    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import torch.fx shape propagation utilities: {e}") from e

    gm.eval()
    with torch.no_grad():
        ShapeProp(gm).propagate(example_input)

    # Map shapes back to our collected ops.
    # - For call_module: match by module qualname (n.target).
    # - For functional nodes we named as fx_{n.name}: match by that derived name.
    in_shape_by_name: Dict[str, Tuple[int, ...]] = {}
    out_shape_by_name: Dict[str, Tuple[int, ...]] = {}
    for n in gm.graph.nodes:
        key: Optional[str] = None
        if n.op == "call_module":
            key = str(n.target)
        elif n.op in ("call_function", "call_method"):
            key = f"fx_{n.name}"
        else:
            continue

        # Output tensor meta
        tm = n.meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape"):
            out_shape_by_name[key] = tuple(int(d) for d in tm.shape)

        # Input tensor meta (best-effort): first tensor argument
        in_shape: Optional[Tuple[int, ...]] = None
        for arg in n.args:
            if isinstance(arg, torch.fx.Node):
                tm_in = arg.meta.get("tensor_meta")
                if tm_in is not None and hasattr(tm_in, "shape"):
                    in_shape = tuple(int(d) for d in tm_in.shape)
                    break
        if in_shape is not None:
            in_shape_by_name[key] = in_shape

    for op in collected:
        if op.module_qualname:
            op.in_shape = in_shape_by_name.get(op.module_qualname)
            op.out_shape = out_shape_by_name.get(op.module_qualname)
        else:
            op.in_shape = in_shape_by_name.get(op.name)
            op.out_shape = out_shape_by_name.get(op.name)

    return collected


def _try_extract_ts_tensor_shape(typ: Any) -> Optional[Tuple[int, ...]]:
    """
    Extract tensor sizes from TorchScript node output type if present.
    Works with `TensorType` that has `sizes()` returning a list with optional ints.
    """
    try:
        if hasattr(typ, "sizes"):
            sizes = typ.sizes()
            if sizes is None:
                return None
            dims: List[int] = []
            for d in sizes:
                if d is None:
                    return None
                dims.append(int(d))
            return tuple(dims)
    except Exception:
        return None
    return None


def collect_ops_torchscript(ts: torch.jit.RecursiveScriptModule, example_input: torch.Tensor) -> List[CollectedOp]:
    """
    Collect ops from a TorchScript model.

    Strategy:
    - Get a concrete trace graph from `torch.jit._get_trace_graph` using the example input.
    - Walk nodes in graph order and normalize supported ops (flatten/linear/relu).
    - Capture in/out shapes from node input/output types when available.

    Note: TorchScript graphs may lower `linear` into `aten::addmm`/`aten::matmul` patterns.
    We detect common patterns and map them to RTL_FC.
    """
    ts.eval()
    try:
        trace_graph, _ = torch.jit._get_trace_graph(ts, (example_input,))  # type: ignore[attr-defined]
        graph = trace_graph
    except Exception as e:
        raise RuntimeError(
            "Failed to obtain TorchScript trace graph for mapping. "
            "Try eager (--model-module/--model-class) mode. "
            f"Error: {e}"
        ) from e

    collected: List[CollectedOp] = []

    # Heuristic: map in execution order; create stable names by incrementing counters.
    counters: Dict[str, int] = {"flatten": 0, "linear": 0, "relu": 0, "unsupported": 0}

    for node in graph.nodes():
        kind = node.kind()
        op_kind: str
        extra: Dict[str, Any] = {}

        # Supported:
        if kind in ("aten::flatten",):
            op_kind = "flatten"
        elif kind in ("aten::relu", "aten::relu_", "aten::threshold"):
            op_kind = "relu"
        elif kind in ("aten::linear", "aten::addmm", "aten::mm", "aten::matmul"):
            # Many FCs compile down to addmm/mm/matmul (+ bias add). We'll map addmm/mm/matmul
            # to RTL_FC and validate shapes/params separately.
            op_kind = "linear"
            extra["ts_kind"] = kind
        elif kind in ("aten::_convolution", "aten::conv2d", "aten::max_pool2d", "aten::softmax"):
            op_kind = "unsupported"
            extra["ts_kind"] = kind
        else:
            # ignore pure reshapes? No: requirement says unsupported ops must stop with clear error.
            op_kind = "unsupported"
            extra["ts_kind"] = kind

        counters[op_kind] = counters.get(op_kind, 0) + 1
        name = f"ts_{op_kind}{counters[op_kind]}"

        op = CollectedOp(name=name, op_kind=op_kind, module_qualname=None, extra=extra)

        # Shapes (best-effort): use first tensor input/output type.
        try:
            in_shapes: List[Tuple[int, ...]] = []
            for v in node.inputs():
                shp = _try_extract_ts_tensor_shape(v.type())
                if shp is not None and len(shp) > 0:
                    in_shapes.append(shp)
            out_shapes: List[Tuple[int, ...]] = []
            for v in node.outputs():
                shp = _try_extract_ts_tensor_shape(v.type())
                if shp is not None and len(shp) > 0:
                    out_shapes.append(shp)
            if in_shapes:
                op.in_shape = in_shapes[0]
            if out_shapes:
                op.out_shape = out_shapes[0]
        except Exception:
            pass

        collected.append(op)

    # Filter out nodes that are definitely not meaningful layers? We do NOT filter:
    # if the graph contains unsupported ops, we want to report them.
    return collected


# -----------------------------
# Parameter association (Linear -> weight/bias)
# -----------------------------


@dataclass
class LinearParams:
    weight: np.ndarray  # float32 (out_features, in_features)
    bias: np.ndarray  # float32 (out_features,)
    qualname: str  # used for naming exports


def collect_linear_params_eager(model: nn.Module) -> Dict[str, LinearParams]:
    out: Dict[str, LinearParams] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy().astype(np.float32)
            b = m.bias.detach().cpu().numpy().astype(np.float32) if m.bias is not None else np.zeros((w.shape[0],), np.float32)
            out[name] = LinearParams(weight=w, bias=b, qualname=name)
    return out


def collect_linear_params_generic_from_state_dict(sd: Dict[str, torch.Tensor]) -> List[LinearParams]:
    """
    Collect candidate Linear-like params from a state_dict (TorchScript or eager state_dict).
    Returns a list that can be matched to linear ops by shape.
    """
    weights: List[Tuple[str, np.ndarray]] = []
    biases: Dict[str, np.ndarray] = {}

    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        arr = v.detach().cpu().numpy()
        if k.endswith(".weight") and arr.ndim == 2:
            weights.append((k, arr.astype(np.float32)))
        elif k.endswith(".bias") and arr.ndim == 1:
            biases[k[: -len(".bias")]] = arr.astype(np.float32)

    candidates: List[LinearParams] = []
    for wk, w in weights:
        prefix = wk[: -len(".weight")]
        b = biases.get(prefix, np.zeros((w.shape[0],), np.float32))
        candidates.append(LinearParams(weight=w, bias=b, qualname=prefix.replace(".", "_")))

    # Deterministic order: sort by qualname to stabilize output names, but mapping uses shape matching.
    candidates.sort(key=lambda p: p.qualname)
    return candidates


def assign_linear_params_to_ops(
    ops: List[CollectedOp],
    eager_model: Optional[nn.Module],
    ts_model: Optional[torch.jit.RecursiveScriptModule],
) -> Dict[str, LinearParams]:
    """
    Return mapping: op.name -> LinearParams for each op_kind == 'linear'.
    For eager: uses module qualname directly when available.
    For TorchScript: matches by (out_features, in_features) shape against state_dict.
    """
    if eager_model is not None:
        by_name = collect_linear_params_eager(eager_model)
        out: Dict[str, LinearParams] = {}
        for op in ops:
            if op.op_kind != "linear":
                continue
            if op.module_qualname and op.module_qualname in by_name:
                out[op.name] = by_name[op.module_qualname]
            else:
                # FX functional linear without module reference is not supported in this RTL mapper.
                raise RuntimeError(
                    f"Found a 'linear' op ({op.name}) without an associated nn.Linear module. "
                    "Please rewrite the model to use nn.Linear modules, or export as TorchScript and retry."
                )
        return out

    if ts_model is None:
        raise ValueError("Either eager_model or ts_model must be provided.")

    sd = ts_model.state_dict()
    candidates = collect_linear_params_generic_from_state_dict(sd)
    unused = candidates.copy()

    out: Dict[str, LinearParams] = {}
    for op in ops:
        if op.op_kind != "linear":
            continue
        if op.out_shape is None or op.in_shape is None:
            raise RuntimeError(
                f"TorchScript linear op '{op.name}' is missing in/out shapes; cannot associate weights. "
                "Try providing a simpler TorchScript model or use eager mode with nn.Linear modules."
            )
        # We expect (N, out_features) output and (N, in_features) input for FC.
        if len(op.in_shape) < 2 or len(op.out_shape) < 2:
            raise RuntimeError(f"Linear op '{op.name}' has unexpected shapes in={op.in_shape}, out={op.out_shape}")
        in_features = int(op.in_shape[-1])
        out_features = int(op.out_shape[-1])
        match_idx = None
        for i, cand in enumerate(unused):
            if cand.weight.shape == (out_features, in_features) and cand.bias.shape == (out_features,):
                match_idx = i
                break
        if match_idx is None:
            raise RuntimeError(
                f"Could not find matching weight/bias for linear op '{op.name}' with "
                f"(out,in)=({out_features},{in_features}). Available weights: {[c.weight.shape for c in unused]}"
            )
        out[op.name] = unused.pop(match_idx)

    return out


# -----------------------------
# Mapping + validation + export
# -----------------------------


def validate_supported_ops(ops: List[CollectedOp]) -> None:
    unsupported = [op for op in ops if op.op_kind == "unsupported"]
    if not unsupported:
        return
    # Raise with the first offending op but also include a small summary.
    lines = ["Unsupported operations detected:"]
    for op in unsupported[:20]:
        lines.append(f"- {op.name}: {op.extra}")
    if len(unsupported) > 20:
        lines.append(f"... and {len(unsupported) - 20} more.")
    lines.append(
        "Only Flatten / Linear / ReLU are supported initially. "
        "Conv2d/MaxPool2d/Softmax are detected but not implemented."
    )
    raise NotImplementedError("\n".join(lines))


def validate_shapes_chain(ops: List[CollectedOp]) -> List[str]:
    """
    Validate that shapes are present and consistent across the sequential chain.
    Returns warnings (non-fatal); fatal inconsistencies raise.
    """
    warnings: List[str] = []
    prev_out: Optional[Tuple[int, ...]] = None
    for op in ops:
        if op.in_shape is None or op.out_shape is None:
            raise RuntimeError(
                f"Missing shape info for op '{op.name}' ({op.op_kind}). "
                "Provide a valid --input-shape and ensure forward pass succeeds."
            )
        if prev_out is not None and tuple(op.in_shape) != tuple(prev_out):
            # Allow batch dimension changes? In sequential nets batch is constant; be strict.
            raise RuntimeError(
                f"Shape mismatch between ops: previous out={prev_out} but '{op.name}' in={op.in_shape}"
            )
        prev_out = op.out_shape

        if op.op_kind == "flatten":
            # For MNIST we expect (N, 784) after flatten.
            if len(op.out_shape) != 2:
                warnings.append(f"Flatten '{op.name}' produced non-2D output shape {op.out_shape}")
        elif op.op_kind == "linear":
            if len(op.in_shape) != 2 or len(op.out_shape) != 2:
                raise RuntimeError(f"Linear '{op.name}' expects 2D (N,features) tensors, got in={op.in_shape} out={op.out_shape}")
        elif op.op_kind == "relu":
            # shape preserved
            pass
    return warnings


def map_to_rtl_netlist(
    ops: List[CollectedOp],
    linear_params_by_opname: Dict[str, LinearParams],
    out_dir: Path,
    scale: int,
    transpose_fc_weights: bool,
) -> Tuple[List[RtlNode], List[str]]:
    """
    Convert collected ops into ordered RTL nodes; export `.mem` for FC layers.
    Returns (nodes, warnings).
    """
    nodes: List[RtlNode] = []
    warnings: List[str] = []

    params_dir = out_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    prev_signal = "in0"
    for idx, op in enumerate(ops):
        op_name = _normalize_activation_name(op.name)
        out_signal = f"n{idx}_out"

        if op.op_kind == "flatten":
            spec = RTL_LIBRARY["flatten"]
            node = RtlNode(
                name=op_name,
                op_type=spec.name,
                inputs=[prev_signal],
                outputs=[out_signal],
                in_shape=_shape_to_list(op.in_shape),
                out_shape=_shape_to_list(op.out_shape),
                params={},
            )
        elif op.op_kind == "relu":
            spec = RTL_LIBRARY["relu"]
            node = RtlNode(
                name=op_name,
                op_type=spec.name,
                inputs=[prev_signal],
                outputs=[out_signal],
                in_shape=_shape_to_list(op.in_shape),
                out_shape=_shape_to_list(op.out_shape),
                params={},
            )
        elif op.op_kind == "linear":
            spec = RTL_LIBRARY["linear"]
            if op.name not in linear_params_by_opname:
                raise RuntimeError(f"Internal error: missing params for linear op '{op.name}'.")

            lp = linear_params_by_opname[op.name]
            w_i16 = float_to_int16(lp.weight, scale=scale)
            b_i16 = float_to_int16(lp.bias, scale=scale)

            if transpose_fc_weights:
                # Some RTL FC blocks expect weight matrix as (in_features, out_features)
                w_i16_to_write = w_i16.T
            else:
                w_i16_to_write = w_i16

            in_features = int(lp.weight.shape[1])
            out_features = int(lp.weight.shape[0])

            # Validate dimensions vs shapes.
            if op.in_shape is None or op.out_shape is None:
                raise RuntimeError(f"Missing shapes for linear op '{op.name}'")
            if int(op.in_shape[-1]) != in_features:
                raise RuntimeError(
                    f"Linear '{op.name}' input features mismatch: op.in_shape[-1]={op.in_shape[-1]} "
                    f"but weight expects in_features={in_features}"
                )
            if int(op.out_shape[-1]) != out_features:
                raise RuntimeError(
                    f"Linear '{op.name}' output features mismatch: op.out_shape[-1]={op.out_shape[-1]} "
                    f"but weight expects out_features={out_features}"
                )

            w_mem = params_dir / f"{lp.qualname}_weights.mem"
            b_mem = params_dir / f"{lp.qualname}_biases.mem"
            write_mem_hex_int16(w_mem, w_i16_to_write)
            write_mem_hex_int16(b_mem, b_i16)

            # Validate `.mem` element count.
            expected_w_elems = int(w_i16_to_write.size)
            expected_b_elems = int(b_i16.size)
            if expected_w_elems != int(w_i16.size):  # transpose doesn't change count
                raise AssertionError("Unexpected weight elem count mismatch after transpose.")
            if expected_b_elems != out_features:
                raise RuntimeError(f"Bias element count mismatch for '{op.name}': {expected_b_elems} != {out_features}")

            node = RtlNode(
                name=op_name,
                op_type=spec.name,
                inputs=[prev_signal],
                outputs=[out_signal],
                in_shape=_shape_to_list(op.in_shape),
                out_shape=_shape_to_list(op.out_shape),
                params={
                    "weight_mem": str(w_mem.relative_to(out_dir)).replace("\\", "/"),
                    "bias_mem": str(b_mem.relative_to(out_dir)).replace("\\", "/"),
                    "in_features": in_features,
                    "out_features": out_features,
                    "scale": scale,
                    "transpose_weights": bool(transpose_fc_weights),
                },
            )
        else:
            raise NotImplementedError(f"Unsupported op_kind '{op.op_kind}' at '{op.name}'")

        nodes.append(node)
        prev_signal = out_signal

    return nodes, warnings


def write_netlist_json(out_dir: Path, nodes: List[RtlNode]) -> Path:
    out_path = out_dir / "netlist.json"
    payload = [asdict(n) for n in nodes]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_mapping_report(
    out_dir: Path,
    input_shape: Tuple[int, ...],
    ops: List[CollectedOp],
    nodes: List[RtlNode],
    warnings: List[str],
    scale: int,
    transpose_fc_weights: bool,
    mode_desc: str,
) -> Path:
    out_path = out_dir / "mapping_report.txt"
    lines: List[str] = []
    lines.append("RTL Mapping Report")
    lines.append("==================")
    lines.append("")
    lines.append(f"Mode: {mode_desc}")
    lines.append(f"Input shape: {list(input_shape)}")
    lines.append(f"Scale factor: {scale}")
    lines.append(f"Transpose FC weights: {transpose_fc_weights}")
    lines.append("")
    lines.append("Layer coverage:")
    counts: Dict[str, int] = {}
    for op in ops:
        counts[op.op_kind] = counts.get(op.op_kind, 0) + 1
    for k in sorted(counts):
        lines.append(f"- {k}: {counts[k]}")
    lines.append("")
    lines.append("Ordered mapping:")
    for i, n in enumerate(nodes):
        lines.append(
            f"{i:02d}. {n.name}  {n.op_type}  in={n.in_shape} -> out={n.out_shape}  params={list(n.params.keys())}"
        )
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_sv_wrapper(out_dir: Path, nodes: List[RtlNode]) -> Path:
    """
    Write a simple SystemVerilog wrapper skeleton that instantiates blocks in sequence.
    This is intentionally a skeleton: exact ports/handshake depend on your RTL library.
    """
    out_path = out_dir / "rtl_wrapper.sv"
    lines: List[str] = []
    lines.append("// Auto-generated RTL wrapper skeleton by rtl_mapper.py")
    lines.append("// NOTE: This is a skeleton. Update ports/handshake to match your RTL blocks.")
    lines.append("")
    lines.append("module rtl_wrapper (")
    lines.append("  input  logic        clk,")
    lines.append("  input  logic        rst_n,")
    lines.append("  input  logic        in_valid,")
    lines.append("  input  logic [15:0] in_data,   // TODO: width/packing for your input")
    lines.append("  output logic        out_valid,")
    lines.append("  output logic [15:0] out_data   // TODO: width/packing for your output")
    lines.append(");")
    lines.append("")
    lines.append("  // Internal wires (TODO: define correct widths and buses)")
    lines.append("  logic        v0;")
    lines.append("  logic [15:0] s0;")
    lines.append("")
    lines.append("  // Example chaining assumes each block consumes/produces streaming samples.")
    lines.append("  // Replace with your actual interface (AXI-stream, valid/ready, etc).")
    lines.append("")
    lines.append("  // Input assignment")
    lines.append("  assign v0 = in_valid;")
    lines.append("  assign s0 = in_data;")
    lines.append("")
    prev_v = "v0"
    prev_s = "s0"
    for i, n in enumerate(nodes):
        v = f"v{i+1}"
        s = f"s{i+1}"
        lines.append(f"  logic        {v};")
        lines.append(f"  logic [15:0] {s};  // TODO: adjust width/bus for {n.op_type}")
        inst = f"u_{n.name}"
        if n.op_type == "RTL_FLATTEN":
            lines.append(f"  RTL_FLATTEN {inst} (/* TODO ports */);")
        elif n.op_type == "RTL_RELU":
            lines.append(f"  RTL_RELU {inst} (/* TODO ports */);")
        elif n.op_type == "RTL_FC":
            lines.append(f"  // params: weight={n.params.get('weight_mem')} bias={n.params.get('bias_mem')}")
            lines.append(f"  RTL_FC {inst} (/* TODO ports */);")
        else:
            lines.append(f"  // Unsupported node type in wrapper: {n.op_type}")
        lines.append("")
        prev_v, prev_s = v, s

    lines.append("  assign out_valid = " + prev_v + ";")
    lines.append("  assign out_data  = " + prev_s + ";")
    lines.append("endmodule")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# -----------------------------
# Fixed-point "RTL-sim" reloader (self-test / sanity)
# -----------------------------


def read_mem_hex_int16(path: Path) -> np.ndarray:
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    vals = []
    for line in txt:
        line = line.strip()
        if not line:
            continue
        u16 = int(line, 16) & 0xFFFF
        # convert to signed int16
        if u16 >= 0x8000:
            u16 -= 0x10000
        vals.append(u16)
    return np.array(vals, dtype=np.int16)


class FixedPointRtlSim(nn.Module):
    """
    A minimal Python model that mirrors the exported RTL chain:
    - Applies quantize_tensor at boundaries similar to the reference script.
    - Uses float tensors derived from exported int16 params (divided by scale).
    """

    def __init__(self, nodes: List[RtlNode], out_dir: Path):
        super().__init__()
        self.nodes = nodes
        self.out_dir = out_dir

        # Preload FC params
        self.fc_params: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]] = {}
        for n in nodes:
            if n.op_type != "RTL_FC":
                continue
            w_path = out_dir / Path(n.params["weight_mem"])
            b_path = out_dir / Path(n.params["bias_mem"])
            scale = int(n.params.get("scale", 256))
            w_i16 = read_mem_hex_int16(w_path)
            b_i16 = read_mem_hex_int16(b_path)

            out_features = int(n.params["out_features"])
            in_features = int(n.params["in_features"])
            transpose = bool(n.params.get("transpose_weights", False))

            if transpose:
                w_i16 = w_i16.reshape(in_features, out_features).T.reshape(-1)
            w_i16 = w_i16.reshape(out_features, in_features)
            b_i16 = b_i16.reshape(out_features)

            w_f = torch.tensor(int16_to_float(w_i16, scale), dtype=torch.float32)
            b_f = torch.tensor(int16_to_float(b_i16, scale), dtype=torch.float32)
            self.fc_params[n.name] = (w_f, b_f, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x
        for n in self.nodes:
            if n.op_type == "RTL_FLATTEN":
                t = torch.flatten(t, 1)
                t = quantize_tensor(t, scale=256)  # consistent boundary quantization
            elif n.op_type == "RTL_RELU":
                t = quantize_tensor(F.relu(t), scale=256)
            elif n.op_type == "RTL_FC":
                w, b, scale = self.fc_params[n.name]
                t = quantize_tensor(F.linear(t, w, b), scale=scale)
            else:
                raise NotImplementedError(f"RTL sim does not support node type: {n.op_type}")
        return t


# -----------------------------
# Self-test
# -----------------------------


class SmallMNISTNet(nn.Module):
    """Reference MNIST FC network (matches QuantizedMNISTNet (1).py)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_self_test(out_dir: Path, scale: int, transpose_fc_weights: bool, quick_samples: int = 1000) -> None:
    """
    Train a tiny MNIST FC model briefly, export RTL mapping, reload params, and evaluate on MNIST test set.
    """
    try:
        from torchvision import datasets, transforms  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Self-test requires torchvision for MNIST loading. Install it with:\n"
            "  pip install torchvision\n"
            f"Original import error: {e}"
        ) from e

    device = torch.device("cpu")
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(out_dir / "mnist_data"), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(out_dir / "mnist_data"), train=False, download=True, transform=transform)

    # Quick training on a subset for speed.
    train_subset = torch.utils.data.Subset(train_ds, list(range(10_000)))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_subset = torch.utils.data.Subset(test_ds, list(range(quick_samples)))
    test_loader = DataLoader(test_subset, batch_size=256, shuffle=False)

    model = SmallMNISTNet().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    for epoch in range(2):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()

    model.eval()

    # Export mapping using the same pipeline as CLI eager mode.
    example_input = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    ops = collect_ops_eager_fx(model, example_input)
    validate_supported_ops(ops)
    chain_warnings = validate_shapes_chain(ops)
    lin_params = assign_linear_params_to_ops(ops, eager_model=model, ts_model=None)
    nodes, map_warnings = map_to_rtl_netlist(ops, lin_params, out_dir, scale, transpose_fc_weights)
    write_netlist_json(out_dir, nodes)
    write_mapping_report(
        out_dir=out_dir,
        input_shape=tuple(example_input.shape),
        ops=ops,
        nodes=nodes,
        warnings=chain_warnings + map_warnings,
        scale=scale,
        transpose_fc_weights=transpose_fc_weights,
        mode_desc="self-test (eager+fx)",
    )

    # Reload exported params and run fixed-point sim.
    sim = FixedPointRtlSim(nodes, out_dir).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            out = sim(images)
            pred = out.argmax(dim=1)
            total += int(labels.numel())
            correct += int((pred.cpu() == labels).sum().item())

    acc = 100.0 * correct / max(1, total)
    print(f"[self-test] Fixed-point RTL-sim accuracy on {total} MNIST test samples: {acc:.2f}%")


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Map PyTorch models to a fixed RTL component library.")
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--torchscript", type=str, default=None, help="Path to TorchScript model (.pt/.ts).")
    mode.add_argument("--model-module", type=str, default=None, help="Path to Python module (.py) defining the model class.")
    p.add_argument("--model-class", type=str, default=None, help="Class name inside --model-module (nn.Module).")
    p.add_argument("--model-path", type=str, default=None, help="Optional .pth state_dict for eager model.")

    p.add_argument("--input-shape", type=str, default="1,1,28,28", help='Example input shape, e.g. "1,1,28,28".')
    p.add_argument("--scale", type=int, default=256, help="Fixed-point scale factor (default 256).")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for netlist/params/report.")
    p.add_argument("--transpose-fc-weights", action="store_true", help="Export FC weight matrices transposed.")
    p.add_argument("--emit-wrapper-sv", action="store_true", help="Emit rtl_wrapper.sv skeleton.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    p.add_argument("--self-test", action="store_true", help="Run built-in mapping+RTL-sim self-check on MNIST.")
    return p


def _configure_logging(verbose: bool) -> None:
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=lvl, format="%(levelname)s: %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        run_self_test(out_dir=out_dir, scale=int(args.scale), transpose_fc_weights=bool(args.transpose_fc_weights))
        return 0

    scale = int(args.scale)
    if scale <= 0:
        raise ValueError("--scale must be positive.")

    input_shape = parse_input_shape(args.input_shape)
    example_input = torch.zeros(input_shape, dtype=torch.float32)

    eager_model: Optional[nn.Module] = None
    ts_model: Optional[torch.jit.RecursiveScriptModule] = None

    if args.torchscript:
        ts_model = load_torchscript(Path(args.torchscript))
        mode_desc = f"torchscript ({Path(args.torchscript).name})"
        ops = collect_ops_torchscript(ts_model, example_input)
    else:
        if not args.model_module or not args.model_class:
            raise ValueError(
                "You must provide either --torchscript OR both --model-module and --model-class "
                "(unless using --self-test)."
            )
        eager_model = load_model_from_module(Path(args.model_module), args.model_class)
        maybe_load_state_dict(eager_model, Path(args.model_path) if args.model_path else None)
        eager_model.eval()
        mode_desc = f"eager+fx ({Path(args.model_module).name}:{args.model_class})"
        ops = collect_ops_eager_fx(eager_model, example_input)

    LOGGER.info("Collected %d ops for mapping.", len(ops))

    # Stop immediately on unsupported ops with a clear error.
    validate_supported_ops(ops)

    # Validate shapes across the chain.
    chain_warnings = validate_shapes_chain(ops)

    # Associate Linear ops with weights/bias.
    lin_params = assign_linear_params_to_ops(ops, eager_model=eager_model, ts_model=ts_model)

    # Map to RTL nodes + export params.
    nodes, map_warnings = map_to_rtl_netlist(
        ops=ops,
        linear_params_by_opname=lin_params,
        out_dir=out_dir,
        scale=scale,
        transpose_fc_weights=bool(args.transpose_fc_weights),
    )

    netlist_path = write_netlist_json(out_dir, nodes)
    report_path = write_mapping_report(
        out_dir=out_dir,
        input_shape=input_shape,
        ops=ops,
        nodes=nodes,
        warnings=chain_warnings + map_warnings,
        scale=scale,
        transpose_fc_weights=bool(args.transpose_fc_weights),
        mode_desc=mode_desc,
    )
    LOGGER.info("Wrote netlist: %s", netlist_path)
    LOGGER.info("Wrote report:  %s", report_path)
    LOGGER.info("Wrote params to: %s", (out_dir / "params"))

    if args.emit_wrapper_sv:
        sv_path = write_sv_wrapper(out_dir, nodes)
        LOGGER.info("Wrote SV wrapper skeleton: %s", sv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

