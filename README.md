# <center>Speech-To-Image (STT) Stable Diffusion WebUI implementation for SUTD AI Megacentre</center>

## Adapted from [sygil-webui](https://github.com/Sygil-Dev/sygil-webui), originally created by [Sygil.Dev](https://github.com/sygil-dev)

## Installation steps for Linux

- Install Anaconda/Minconda
- Clone <https://github.com/Inc0mple/sygil-webui>
- Run `installer/install.sh`
- Activate conda environment (ldm) with `conda activate ldm`
- Install the following libraries:
  - `conda install PyAudio`
  - `conda install git+<https://github.com/Uberi/speech_recognition>`
  - `conda install tf-nightly`
  - `conda install vosk`
- Download and unzip [AI Megacentre Additional Files.7z](https://drive.google.com/file/d/1axXDdrIPYjWn_CbDFODdhjE2e3MAKO44/view?usp=sharing) which contain the STT widget, and the models for Image Generation and Speech-To-Text.
- Copy the contents of the unzipped folder into sygil-webui, merging the `models` and `scripts` folder.

## Run steps for Linux

- Run `./webui.sh`
- Press enter and select streamlit

## Installation/Run steps for Windows

Same as the Linux instructions above, except run `installer/install.bat` and `./webui.bat` instead of their `.sh` variant

## Features

- **Simplified and streamlined UI for general audience**
  - The original Web-UI for stable diffusion is cluttered and full of settings and options that an average user may find intimiating to use and navigate. This implementation reduces the UI elements to the bare essentials for ease of use.
- **Speech-To-Image and Text-To-Image functionality**
  - Users have the option to either type the prompts or use the speech-to-text functionality for image generation using the Stable Diffusion v1.5 model.
- **Automatic image upscalling**
  - Images are originally generated in a relatively low 512x512 resolution to save on computational resource, before being automatically scaled up to 900x900 by the  RealESRGAN_x4plus image upscaler.
- **Automatic prompt extension for addition of suitable style cues**
  - A layperson might not know how to engineer a prompt for their intended image. Moreover, it is impossible to add seperate tags and cumbersome to be detailed and specific when using speech to describe their image. [Prompt extension](https://github.com/daspartho/prompt-extend) works by predicting appropriate tags to add to a given simple prompt. This increases ease of use for the layperson, and usually results in images that are more aesthetically pleasing and in line with the user's intent.
- **Rudimentary NSFW filter (not 100%)**
  - It is difficult to predict the image output of an input text prompt, as well as predict the tags added by the prompt extender. Some questionable images may be unintentionally generated from seemingly innocuous prompts. Hence, "negative prompts" are built into the app to filter out potentially NSFW images. However, this solution is not perfect, and some NSFW images may still make it through the filter.
