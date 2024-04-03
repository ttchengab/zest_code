# ___***ZeST: Zero-Shot Material Transfer from a Single Image***___
This is the official implementation of ZeST: Zero-Shot Material Transfer from a Single Image.

## Installation
This work is built from the IP-Adaptor. Please follow the following installation instructions to get IP-Adapter for Stable Diffusion XL ready.

We begin by installing the diffusers library:

```
pip install diffusers==0.22.1
```

Then clone this repo:

```
https://github.com/ttchengab/zest_code.git
```

Then install IP Adaptor and download the needed models:
```
# install ip-adapter
cd zest_code
git clone https://github.com/tencent-ailab/IP-Adapter.git
mv IP-Adapter/ip_adapter ip_adapter
rm -r IP-Adapter/

# download the models
cd IP-Adapter
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
```

## Download Models

You can download models from [here](https://huggingface.co/h94/IP-Adapter). To run the demo, you should also download the following models:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [SG161222/Realistic_Vision_V4.0_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V4.0_noVAE)
- [ControlNet models](https://huggingface.co/lllyasviel)

## Demo on Single Image

After installation and downloading the models, you can then use `demo.ipynb` to perform material transfer from a single image and material exemplar

## Inferencing on batch of images
To cross-inference on a set of input images and material exemplars, first create the following directory: 

```
mkdir demo_assets/output_images
```


