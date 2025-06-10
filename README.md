# Convert diffusion (Flux, SD, etc) safetensors to GGUF

Example usage:

```sh
# prepare
pip install gguf torch

# example model from https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main
python convert_diffusion_to_gguf.py ../models/FLUX.1-dev/flux1-dev.safetensors --arch flux --outtype Q4_0
# output file: model-Q4_0.gguf

# to view help: python convert_diffusion_to_gguf.py -h
```

TODO: `Qx_K` quants currently not implemented in python, we will need to use `libggml`
