# DiPTeR - Differentiable Procedural Texture Renderer

<<<<<<< HEAD
DiPTer is a framework for creating procedurally generated textures that are fully differentiable. It features a differentiable renderer specialized in procedural textures implemented in PyTorch and a graphical interface, including a node editor, implemented using PyQt5. DiPTeR's main feature however, is a parameter estimation tool that lets users automatically estimate parameter values of a procedural texture model, based on an input target texture.

This project was created as part of my master thesis.

![Image of DiPTeR main interface](https://imgur.com/TdTj4Us.png)

## Parameter Estimation

The parameter estimation algorithm uses gradient descent and loss functions to minimize the difference between a user input image and the output of a user designed procedural texture.

![Gif of parameter estimation](https://i.imgur.com/BsE4Oxz.gif)
=======
DiPTer is a tool to create procedurally generated textures that are fully differentiable.
 
 :warning:**This project is in β phase. Expect nothing to work!**:warning:
 
>>>>>>> 92d5236c261404282eae86c7a38b636da76f6c24
 
## Setup

### Install dependencies :warning: NOT UPDATED YET :warning:

Python dependencies can be installed from the `environment.yml` file using conda. However, I've had problems installing from this format before, so as an alternative, install
 from the `env.txt` file instead using `conda env create --file env.txt`.
  
### Glumpy and freetype
In order for `glumpy` to work, `freetype` is needed. To install it on Windows, you can download the `freetype.dll` from 
[HERE](https://github.com/ubawurinna/freetype-windows-binaries/blob/master/win64/freetype.dll). You then need to set the `FT_Library_filename` variable in `glumpy/ext/freetype
/__init__.py` to point to this `.dll`...

For Anaconda users, this folder is usually located in `C:/Users/<username>/<Anaconda3 or miniconda3>/envs/DiPTer/Lib/site-packages`
