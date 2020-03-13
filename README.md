# DiPTer - Differentiable Procedural Texture Renderer

DiPTer is a tool to create procedurally generated textures that are fully differentiable.
 
 :warning:**This project is in very early alpha and has just been started. Expect nothing to work!**:warning:
 
 
## Contribute

### Glumpy and freetype
In order for `glumpy` to work, `freetype` is needed. To install it on Windows, you can download the `freetype.dll` from 
[HERE](https://github.com/ubawurinna/freetype-windows-binaries/blob/master/win64/freetype.dll). You then need to set the `FT_Library_filename` in `glumpy/ext/freetype/__init__
.py` to point to this `.dll`...
