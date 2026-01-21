"""
convert_model_to_rtl.py
=======================

Utility script that maps a simple PyTorch feed‑forward model (e.g. the
`SmallMNISTNet` defined in `QuantizedMNISTNet (1).py`) to the RTL component
library in `to_rtl/`. It wraps `rtl_mapper.py` so you can run one command to:

- load the Python model + optional state_dict
- collect ops with torch.fx
- quantize weights/biases to int16
- emit `.mem` parameter files, `netlist.json`, `mapping_report.txt`
- optionally emit a `rtl_wrapper.sv` skeleton and copy the RTL templates

Example:
    python convert_model_to_rtl.py ^
        --model-module "QuantizedMNISTNet (1).py" ^
        --model-class SmallMNISTNet ^
        --model-path mnist_model.pth ^
        --out-dir ./rtl_build ^
        --copy-rtl-templates ^
        --emit-wrapper-sv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional, Sequence

import torch

import rtl_mapper as rm


def _copy_rtl_templates(src_dir: Path, dst_dir: Path) -> None:
    """Copy the RTL building blocks (SystemVerilog) into the output."""
    if not src_dir.exists():
        raise FileNotFoundError(f"RTL template directory not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for sv_file in src_dir.glob("*.sv"):
        shutil.copy2(sv_file, dst_dir / sv_file.name)


def run_mapping(
    model_module: Path,
    model_class: str,
    model_path: Optional[Path],
    input_shape: Sequence[int],
    out_dir: Path,
    scale: int = 256,
    transpose_fc_weights: bool = False,
    emit_wrapper_sv: bool = False,
    copy_rtl_templates: bool = False,
    rtl_template_dir: Optional[Path] = None,
) -> None:
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    example_input = torch.zeros(tuple(int(d) for d in input_shape), dtype=torch.float32)

    # Load eager model and optional weights.
    model = rm.load_model_from_module(model_module, model_class)
    rm.maybe_load_state_dict(model, model_path)
    model.eval()

    # Collect ops and validate.
    ops = rm.collect_ops_eager_fx(model, example_input)
    rm.validate_supported_ops(ops)
    chain_warnings = rm.validate_shapes_chain(ops)

    # Map linear ops to weights/bias.
    lin_params = rm.assign_linear_params_to_ops(ops, eager_model=model, ts_model=None)

    # Export RTL mapping artifacts.
    nodes, map_warnings = rm.map_to_rtl_netlist(
        ops=ops,
        linear_params_by_opname=lin_params,
        out_dir=out_dir,
        scale=scale,
        transpose_fc_weights=transpose_fc_weights,
    )

    rm.write_netlist_json(out_dir, nodes)
    rm.write_mapping_report(
        out_dir=out_dir,
        input_shape=tuple(example_input.shape),
        ops=ops,
        nodes=nodes,
        warnings=chain_warnings + map_warnings,
        scale=scale,
        transpose_fc_weights=transpose_fc_weights,
        mode_desc=f"eager+fx ({model_module.name}:{model_class})",
    )

    if emit_wrapper_sv:
        rm.write_sv_wrapper(out_dir, nodes)

    if copy_rtl_templates:
        template_dir = rtl_template_dir or (Path(__file__).parent / "to_rtl")
        _copy_rtl_templates(template_dir, out_dir / "rtl_lib")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert a PyTorch FC model to RTL artifacts.")
    p.add_argument("--model-module", type=str, required=True, help="Path to Python file defining the model class.")
    p.add_argument("--model-class", type=str, required=True, help="Class name (nn.Module) inside the module.")
    p.add_argument("--model-path", type=str, default=None, help="Optional .pth state_dict to load before export.")
    p.add_argument("--input-shape", type=str, default="1,1,28,28", help='Example input shape, e.g. \"1,1,28,28\".')
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for RTL artifacts.")
    p.add_argument("--scale", type=int, default=256, help="Fixed-point scale factor for quantization.")
    p.add_argument("--transpose-fc-weights", action="store_true", help="Export FC weights as (in, out) instead of (out, in).")
    p.add_argument("--emit-wrapper-sv", action="store_true", help="Also emit rtl_wrapper.sv skeleton.")
    p.add_argument("--copy-rtl-templates", action="store_true", help="Copy SystemVerilog blocks from ./to_rtl into out-dir/rtl_lib.")
    p.add_argument("--rtl-template-dir", type=str, default=None, help="Override source directory for RTL templates (defaults to ./to_rtl).")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_shape = rm.parse_input_shape(args.input_shape)
    run_mapping(
        model_module=Path(args.model_module),
        model_class=args.model_class,
        model_path=Path(args.model_path) if args.model_path else None,
        input_shape=input_shape,
        out_dir=Path(args.out_dir),
        scale=int(args.scale),
        transpose_fc_weights=bool(args.transpose_fc_weights),
        emit_wrapper_sv=bool(args.emit_wrapper_sv),
        copy_rtl_templates=bool(args.copy_rtl_templates),
        rtl_template_dir=Path(args.rtl_template_dir) if args.rtl_template_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
