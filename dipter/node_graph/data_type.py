import enum


# ----- Internal Types --------


class DataType(enum.Enum):
    Float = "type_float"
    Int = "type_int"
    Int_Choice = "type_int_choice"
    Vec3_Float = "type_vec3_float"
    Vec3_RGB = "type_vec3_rgb"
    Shader = "type_shader"
    String = "type_string"


def dtype_size(dtype: DataType) -> int:
    if "vec3" in dtype.value:
        return 3
    else:
        return 1


def is_vector(dtype: DataType) -> bool:
    return dtype_size(dtype) > 1
