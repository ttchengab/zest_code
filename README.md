# ___***ZeST: Zero-Shot Material Transfer from a Single Image***___

## Installation
This work is built from the IP-Adaptor. Please follow the following installation instructions to get IP-Adapter for Stable Diffusion XL ready.

```
# install latest diffusers
pip install diffusers==0.22.1

# install ip-adapter
pip install git+https://github.com/tencent-ailab/IP-Adapter.git

# download the models
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models

# then you can use the notebook
```

## Download Models

you can download models from [here](https://huggingface.co/h94/IP-Adapter). To run the demo, you should also download the following models:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE)
- [ControlNet models](https://huggingface.co/lllyasviel)
