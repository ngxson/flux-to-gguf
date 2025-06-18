#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import argparse
import json
import safetensors.torch
import os
import sys
from pathlib import Path
from hashlib import sha256
from typing import Any, ContextManager, cast
from torch import Tensor

import numpy as np
import torch
import gguf

# TODO: add more:
SUPPORTED_ARCHS = ["flux", "sd3", "ltxv", "hyvid", "wan", "hidream"]

logger = logging.getLogger(__name__)

class QuantConfig():
    ftype: gguf.LlamaFileType
    qtype: gguf.GGMLQuantizationType

    def __init__(self, ftype: gguf.LlamaFileType, qtype: gguf.GGMLQuantizationType):
        self.ftype = ftype
        self.qtype = qtype


qconfig_map: dict[str, QuantConfig] = {
    "F16": QuantConfig(gguf.LlamaFileType.MOSTLY_F16, gguf.GGMLQuantizationType.F16),
    "BF16": QuantConfig(gguf.LlamaFileType.MOSTLY_BF16, gguf.GGMLQuantizationType.BF16),
    "Q8_0": QuantConfig(gguf.LlamaFileType.MOSTLY_Q8_0, gguf.GGMLQuantizationType.Q8_0),
    "Q6_K": QuantConfig(gguf.LlamaFileType.MOSTLY_Q6_K, gguf.GGMLQuantizationType.Q6_K),
    "Q5_K_S": QuantConfig(gguf.LlamaFileType.MOSTLY_Q5_K_S, gguf.GGMLQuantizationType.Q5_K),
    "Q5_1": QuantConfig(gguf.LlamaFileType.MOSTLY_Q5_1, gguf.GGMLQuantizationType.Q5_1),
    "Q5_0": QuantConfig(gguf.LlamaFileType.MOSTLY_Q5_0, gguf.GGMLQuantizationType.Q5_0),
    "Q4_K_S": QuantConfig(gguf.LlamaFileType.MOSTLY_Q4_K_S, gguf.GGMLQuantizationType.Q4_K),
    "Q4_1": QuantConfig(gguf.LlamaFileType.MOSTLY_Q4_1, gguf.GGMLQuantizationType.Q4_1),
    "Q4_0": QuantConfig(gguf.LlamaFileType.MOSTLY_Q4_0, gguf.GGMLQuantizationType.Q4_0),
    "Q3_K_S": QuantConfig(gguf.LlamaFileType.MOSTLY_Q3_K_S, gguf.GGMLQuantizationType.Q3_K),
    #"Q2_S": QuantConfig(gguf.LlamaFileType.MOSTLY_Q2_K, gguf.GGMLQuantizationType.Q2_K), # not yet supported in python
}


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy())
        )

    @classmethod
    def meta_with_dtype_and_shape(cls, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(meta=cls.meta_with_dtype_and_shape(dtype, shape), args=(st_slice,), func=lambda s: s[:])
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


class Converter():
    path_safetensors: Path
    endianess: gguf.GGUFEndian
    outtype: QuantConfig
    outfile: Path
    gguf_writer: gguf.GGUFWriter

    def __init__(
        self, 
        arch: str, 
        path_safetensors: Path, 
        endianess: gguf.GGUFEndian, 
        outtype: QuantConfig, 
        outfile: Path,
        subfolder: str = None,
        repo_id: str = None, 
        is_diffusers: bool = False
    ):
        self.path_safetensors = path_safetensors
        self.endianess = endianess
        self.outtype = outtype
        self.outfile = outfile

        self.gguf_writer = gguf.GGUFWriter(path=None, arch=arch, endianess=self.endianess)
        self.gguf_writer.add_file_type(self.outtype.ftype)
        self.gguf_writer.add_type("diffusion") # for HF hub to detect the type correctly
        if repo_id:
            self.gguf_writer.add_string("repo_id", repo_id)
        if subfolder:
            self.gguf_writer.add_string("subfolder", subfolder)
        if is_diffusers:
            self.gguf_writer.add_bool("is_diffusers", True)

        # load tensors and process
        from safetensors import safe_open
        ctx = cast(ContextManager[Any], safe_open(path_safetensors, framework="pt", device="cpu"))
        with ctx as model_part:
            for name in model_part.keys():
                data = model_part.get_slice(name)
                data = LazyTorchTensor.from_safetensors_slice(data)
                self.process_tensor(name, data)


    def process_tensor(self, name: str, data_torch: LazyTorchTensor) -> None:
        is_1d = len(data_torch.shape) == 1
        current_dtype = data_torch.dtype
        target_dtype = gguf.GGMLQuantizationType.F32 if is_1d else self.outtype.qtype

        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        data = data_torch.numpy()

        if current_dtype != target_dtype:
            from custom_quants import quantize as custom_quantize, QuantError
            try:
                data = custom_quantize(data, target_dtype)
            except QuantError as e:
                logger.warning("%s, %s", e, "falling back to F16")
                target_dtype = gguf.GGMLQuantizationType.F16
                data = custom_quantize(data, target_dtype)

        # reverse shape to make it similar to the internal ggml dimension order
        shape = gguf.quant_shape_from_byte_shape(data.shape, target_dtype) if data.dtype == np.uint8 else data.shape
        shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"
        logger.info(f"{f'%-32s' % f'{name},'} {current_dtype} --> {target_dtype.name}, shape = {shape_str}")

        # add tensor to gguf
        self.gguf_writer.add_tensor(name, data, raw_dtype=target_dtype)

    def write(self) -> None:
        self.gguf_writer.write_header_to_file(path=self.outfile)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

# https://github.com/bghira/SimpleTuner/blob/cea2457ab063f6dedb9e697830ae68a96be90641/helpers/training/save_hooks.py#L64
def _merge_sharded_checkpoints(folder: Path):
    with open(folder / "diffusion_pytorch_model.safetensors.index.json", "r") as f:
        ckpt_metadata = json.load(f) 
    weight_map = ckpt_metadata.get("weight_map", None)
    if weight_map is None:
        raise KeyError("'weight_map' key not found in the shard index file.")

    # Collect all unique safetensors files from weight_map
    files_to_load = set(weight_map.values())
    merged_state_dict = {}

    # Load tensors from each unique file
    for file_name in files_to_load:
        part_file_path = folder /  file_name
        if not os.path.exists(part_file_path):
            raise FileNotFoundError(f"Part file {file_name} not found.")

        with safetensors.safe_open(part_file_path, framework="pt", device="cpu") as f:
            for tensor_key in f.keys():
                if tensor_key in weight_map:
                    merged_state_dict[tensor_key] = f.get_tensor(tensor_key)

    return merged_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a flux model to GGUF")
    parser.add_argument(
        "--outfile", type=Path, default=Path("model-{ftype}.gguf"),
        help="path to write to; default: 'model-{ftype}.gguf' ; note: {ftype} will be replaced by the outtype",
    )
    parser.add_argument(
        "--outtype", type=str, choices=qconfig_map.keys(), default="F16",
        help="output quantization scheme",
    )
    parser.add_argument(
        "--arch", type=str, choices=SUPPORTED_ARCHS,
        help="output model architecture",
    )
    parser.add_argument(
        "--bigendian", action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "model", type=Path,
        help="directory containing safetensors model file",
        nargs="?",
    )
    parser.add_argument("--cache_dir", type=Path, help="Directory to store the intermediate files when needed.")
    parser.add_argument("--subfolder", type=Path, default=None, help="Subfolder on the HF Hub to load checkpoints from.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="increase output verbosity",
    )

    args = parser.parse_args()
    if args.model is None:
        parser.error("the following arguments are required: model")
    if args.arch is None:
        parser.error("the following arguments are required: --arch")
    if args.arch not in SUPPORTED_ARCHS:
        parser.error(f"Unsupported architecture: {args.arch}. Supported architectures: {', '.join(SUPPORTED_ARCHS)}")
    return args

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.model.is_dir() and not args.model.is_file():
        if not len(str(args.model).split("/")) == 2:
            logging.error(f"Model path {args.model} does not exist.")
            sys.exit(1)

    is_diffusers = False
    repo_id = None
    if args.model.is_dir():
        logging.info("Supplied a directory.")
        merged_state_dict = None
        files = list(args.model.glob('*.safetensors'))
        n = len(files)
        if n == 0:
            logging.error("No safetensors files found.")
            sys.exit(1)
        if n == 1:
            logging.info(f"Assinging {files[0]} to `args.model`")
            args.model = files[0]
        if n > 1:
            assert args.model / "diffusion_pytorch_model.safetensors.index.json" in list(args.model.glob("*.*"))
            assert args.cache_dir
            merged_state_dict = _merge_sharded_checkpoints(args.model)
            filepath = args.cache_dir / "merged_state_dict.safetensors"
            safetensors.torch.save_file(merged_state_dict, filepath)
            logging.info(f"Serialized merged state dict to {filepath}")
            args.model = Path(filepath)
    
    elif len(str(args.model).split("/")) == 2:
        from huggingface_hub import snapshot_download

        logging.info("Hub repo ID detected.")
        allow_patterns = f"{args.subfolder}/*.*" if args.subfolder else None
        local_dir = snapshot_download(repo_id=str(args.model), local_dir=args.cache_dir, allow_patterns=allow_patterns)
        repo_id = str(args.model)
        local_dir = Path(local_dir)
        local_dir = local_dir / args.subfolder if args.subfolder else local_dir
        merged_state_dict = _merge_sharded_checkpoints(local_dir)
        filepath = args.cache_dir / "merged_state_dict.safetensors" if args.cache_dir else "merged_state_dict.safetensors"
        safetensors.torch.save_file(merged_state_dict, filepath)
        logging.info(f"Serialized merged state dict to {filepath}")
        args.model = Path(filepath)
        is_diffusers = True

    if args.model.suffix != ".safetensors":
        logging.error(f"Model path {args.model} is not a safetensors file.")
        sys.exit(1)

    if args.outfile.suffix != ".gguf":
        logging.error("Output file must have .gguf extension.")
        sys.exit(1)

    qconfig = qconfig_map[args.outtype]
    outfile = Path(str(args.outfile).format(ftype=args.outtype.upper()))

    logger.info(f"Converting model in {args.model} to {outfile} with quantization {args.outtype}")
    converter = Converter(
        arch=args.arch,
        path_safetensors=args.model,
        endianess=gguf.GGUFEndian.BIG if args.bigendian else gguf.GGUFEndian.LITTLE,
        outtype=qconfig,
        outfile=outfile,
        repo_id=repo_id,
        subfolder=str(args.subfolder) if args.subfolder else None,
        is_diffusers=is_diffusers,
    )
    converter.write()
    logger.info(f"Conversion complete. Output written to {outfile}, architecture: {args.arch}, quantization: {qconfig.qtype.name}")

if __name__ == "__main__":
    main()
