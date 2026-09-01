from typing import TypedDict, Literal


class ExecutorException(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class Executer:
    def __init__(self):
        self._done
        self._cancel
        self._result

    def __iter__(self): ...

    def __next__(self): ...

    def done(self): ...

    def cancel(self): ...

    def result(self): ...

    def expense(self): ...

    def exception(self): ...