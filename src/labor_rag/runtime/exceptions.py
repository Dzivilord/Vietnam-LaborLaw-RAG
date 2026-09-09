"""Runtime-level exceptions."""


class RuntimeNotReadyError(RuntimeError):
    pass


class RuntimeExecutionError(RuntimeError):
    pass
