from typing import TypeVar

# T_KEYS = TypeVar("T_KEYS")
T_VAL = TypeVar("T_VAL")


class OrderedTupleDict(dict[tuple, T_VAL]):
    def __setitem__(self, __key: tuple, __value: T_VAL) -> None:
        return super().__setitem__(tuple(sorted(__key)), __value)

    def __getitem__(self, __key: tuple) -> T_VAL:
        return super().__getitem__(tuple(sorted(__key)))


class TwoWayTuple(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __eq__(self, other):
        if super().__eq__(tuple(reversed(other))):
            return True
        else:
            return super().__eq__(self, other)

    def __repr__(self):
        return f"({self[0]}, {self[1]})"
