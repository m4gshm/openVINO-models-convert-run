import openvino as ov
core = ov.Core()
print(core.get_property("NPU", "SUPPORTED_PROPERTIES"))

print(core.get_property("NPU", "NPU_COMPILER_TYPE"))