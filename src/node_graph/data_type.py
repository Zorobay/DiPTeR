import enum


# ----- Internal Types --------
class DataType(enum.Enum):
    Float = "type_float"
    Int = "type_int"
    Vec3_Float = "type_vec3_float"
    Vec3_RGB = "type_vec3_rgb"
    Shader = "type_shader"
