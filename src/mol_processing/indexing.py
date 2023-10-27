class TwoWayTuple(tuple):
    def __new__(cls, *args):
        print(args)
        return super().__new__(cls, args)

    def __eq__(self, other):
        if super().__eq__(tuple(reversed(other))):
            return True
        else:
            return super().__eq__(self, other)
