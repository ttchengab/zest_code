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

#Try with your own material exemplar

Simply place the image into `demo_assets/material_exemplars` and change `texture` variable in `demo.ipynb` to the name of the image.

# Try with your own input image

To use your own input images, we would need to borrow depth predictions using DPT.

Install DPT and its dependencies with:

```
git clone https://github.com/isl-org/DPT.git
pip install -r DPT/requirements.txt
```

Place your images inside `DPT/input/` and obtain the results in `DPT/output/` by running:

```
python DPT/run_monodepth.py
```

Afterwards, place all your files from the `DPT/input/` and `DPT/output/` into `demo_assets/input_imgs` and `demo_assets/depths`, respectively. Change `obj` variable in `demo.ipynb` to the name of the input image.

## Inferencing on batch of images
To cross-inference on a set of input images and material exemplars, first create the following directory: 

```
mkdir demo_assets/output_images
```

Follow the above steps to obtain and put all the material exemplars and corresponding input images/depths into their directories.

Then run:

```
python run_batch.py
```


