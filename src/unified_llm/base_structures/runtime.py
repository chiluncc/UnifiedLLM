from logging import Logger
from typing import TypedDict


# TODO 这个需要等到后面开始做Agent流程图的时候才继续完成，现在有点早了，先写个占位置用，别以后忘记了。


class RuntimeInfo(TypedDict):
    runtime_state: dict
    runtime_config: dict
    runtime_logger: Logger