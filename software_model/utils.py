from typing import List
from llmcompass_utils import size


class DataType:
    def __init__(self, name: str, word_size: int) -> None:
        self.name = name
        self.word_size:int = word_size

data_type_dict = {"int8": DataType("int8", 1), "fp16": DataType("fp16", 2), "fp32": DataType("fp32", 4), "bf16": DataType("bf16", 2)}

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        if not shape or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in shape):
            raise ValueError("Tensor shape must contain positive integers")
        if not isinstance(data_type, DataType):
            raise ValueError("Tensor requires a DataType")
        self.shape = list(shape)
        self.size = size(shape)
        self.data_type = data_type
        
