# Environment notes

Follow the [README setup commands](../README.md#3-setup). Install the selected PyTorch build before `requirements.txt`.

For a GPU requiring CUDA 12.6 instead of 13.0:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

See [PyTorch's installation commands](https://pytorch.org/get-started/previous-versions/#v2100) for supported builds. On macOS, install `torch==2.10.0` from PyPI and use CPU configs; macOS execution has not been validated.

The dependencies include `spectral` for HyperBlood preprocessing and TensorBoard for visualization.
