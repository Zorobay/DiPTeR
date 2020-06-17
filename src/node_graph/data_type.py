import enum


# ----- Internal Types --------
class DataType(enum.Enum):
    ARRAY = "type_array_"
    Float = "type_float"
    Int = "type_int"
    Vec3_Float = ARRAY + "float"
    Vec3_RGB = ARRAY + "rgb"
    Shader = "type_shader"
