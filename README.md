# Convert Flux safetensors to GGUF

Example usage:

```sh
# prepare
pip install gguf torch

# download model from https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main
python convert_flux_to_gguf.py ../models/FLUX.1-dev/flux1-dev.safetensors --outfile model-Q4_0.gguf --outtype Q4_0

# to view help: python convert_flux_to_gguf.py -h
```

TODO: `Qx_K` quants currently not implemented in python, we will need to use `libggml`
