import enum


# ----- Internal Types --------


class DataType(enum.Enum):
    Float = "type_float"
    Int = "type_int"
    Vec3_Float = "type_vec3_float"
    Vec3_RGB = "type_vec3_rgb"
    Shader = "type_shader"


def dtype_size(dtype: DataType):
    if "vec3" in dtype:
        return 3
    else:
        return 1
