import dipter.shaders.lib.glsl_builtins as gl
from dipter.shaders.lib import perlin_noise as pn
from dipter.shaders.lib import vec
from dipter.shaders.shader_super import *
from dipter.shaders.shader_io import ShaderInputParameter, ShaderOutputParameter


class PerlinNoiseShader(FunctionShader):
    FRAGMENT_SHADER_FILENAME = "perlin_noise_shader.glsl"
    FRAGMENT_SHADER_FUNCTION = "cnoise"

    def __init__(self):
        super().__init__()

    def get_inputs(self) -> typing.List[ShaderInputParameter]:
        return [
            ShaderInputParameter("Scale", "scale", dtype=DataType.Float, limits=(0, 1000), default=10),
        ]

    def get_outputs(self) -> typing.List[ShaderOutputParameter]:
        return [
            ShaderOutputParameter("Factor", argument="fac", dtype=DataType.Float),
            ShaderOutputParameter("Color", argument="color", dtype=DataType.Vec3_RGB)
        ]

    def shade_mat(self, scale: Tensor) -> typing.Tuple[Tensor, Tensor]:
        P = Shader.frag_pos() * scale
        Pi0 = torch.floor(P)
        Pi1 = Pi0 + vec.vec3(1.0)
        Pi0 = pn.mod289(Pi0)
        Pi1 = pn.mod289(Pi1)
        Pf0 = gl.fract(P)
        Pf1 = Pf0 - vec.vec3(1.0)
        ix = vec.cat([vec.x(Pi0), vec.x(Pi1), vec.x(Pi0), vec.x(Pi1)])
        iy = vec.cat([vec.yy(Pi0), vec.yy(Pi1)])
        iz0 = vec.zzzz(Pi0)
        iz1 = vec.zzzz(Pi1)

        ixy = pn.permute(pn.permute(ix) + iy)
        ixy0 = pn.permute(ixy + iz0)
        ixy1 = pn.permute(ixy + iz1)

        gx0 = ixy0 * (1.0 / 7.0)
        gy0 = gl.fract(torch.floor(gx0) * (1.0 / 7.0)) - 0.5
        gx0 = gl.fract(gx0)
        gz0 = 0.5 - torch.abs(gx0) - torch.abs(gy0)
        sz0 = gl.step(gz0, 0.0)
        gx0 = gx0 - sz0 * (gl.step(0.0, gx0) - 0.5)
        gy0 = gy0 - sz0 * (gl.step(0.0, gy0) - 0.5)

        gx1 = ixy1 * (1.0 / 7.0)
        gy1 = gl.fract(torch.floor(gx1) * (1.0 / 7.0)) - 0.5
        gx1 = gl.fract(gx1)
        gz1 = 0.5 - torch.abs(gx1) - torch.abs(gy1)
        sz1 = gl.step(gz1, 0.0)
        gx1 = gx1 - sz1 * (gl.step(0.0, gx1) - 0.5)
        gy1 = gy1 - sz1 * (gl.step(0.0, gy1) - 0.5)

        g000 = vec.cat([vec.x(gx0), vec.x(gy0), vec.x(gz0)])
        g100 = vec.cat([vec.y(gx0), vec.y(gy0), vec.y(gz0)])
        g010 = vec.cat([vec.z(gx0), vec.z(gy0), vec.z(gz0)])
        g110 = vec.cat([vec.w(gx0), vec.w(gy0), vec.w(gz0)])
        g001 = vec.cat([vec.x(gx1), vec.x(gy1), vec.x(gz1)])
        g101 = vec.cat([vec.y(gx1), vec.y(gy1), vec.y(gz1)])
        g011 = vec.cat([vec.z(gx1), vec.z(gy1), vec.z(gz1)])
        g111 = vec.cat([vec.w(gx1), vec.w(gy1), vec.w(gz1)])

        norm0 = pn.taylorInvSqrt(vec.cat([gl.dot(g000, g000), gl.dot(g010, g010), gl.dot(g100, g100), gl.dot(g110, g110)]))
        g000 = g000 * vec.x(norm0)
        g010 = g010 * vec.y(norm0)
        g100 = g100 * vec.z(norm0)
        g110 = g110 * vec.w(norm0)
        norm1 = pn.taylorInvSqrt(vec.cat([gl.dot(g001, g001), gl.dot(g011,g011), gl.dot(g101,g101), gl.dot(g111,g111)]))
        g001 = g001 * vec.x(norm1)
        g011 = g011 * vec.y(norm1)
        g101 = g101 * vec.z(norm1)
        g111 = g111 * vec.w(norm1)

        n000 = gl.dot(g000, Pf0)
        n100 = gl.dot(g100, vec.cat([vec.x(Pf1), vec.yz(Pf0)]))
        n010 = gl.dot(g010, vec.cat([vec.x(Pf0), vec.y(Pf1), vec.z(Pf0)]))
        n110 = gl.dot(g110, vec.cat([vec.xy(Pf1), vec.z(Pf0)]))
        n001 = gl.dot(g001, vec.cat([vec.xy(Pf0), vec.z(Pf1)]))
        n101 = gl.dot(g101, vec.cat([vec.x(Pf1), vec.y(Pf0), vec.z(Pf1)]))
        n011 = gl.dot(g011, vec.cat([vec.x(Pf0), vec.yz(Pf1)]))
        n111 = gl.dot(g111, Pf1)

        fade_xyz = pn.fade(Pf0)
        n_z = gl.mix(vec.cat([n000, n100, n010, n110]), vec.cat([n001, n101, n011, n111]), vec.z(fade_xyz))
        n_yz = gl.mix(vec.xy(n_z), vec.zw(n_z), vec.y(fade_xyz))
        n_xyz = gl.mix(vec.x(n_yz), vec.y(n_yz), vec.x(fade_xyz))

        fac = 2.2 * n_xyz
        color = vec.cat([fac, fac, fac])

        return (fac, color)
