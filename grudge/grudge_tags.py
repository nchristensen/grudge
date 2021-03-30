from pytools.tag import Tag, UniqueTag
from meshmode.array_context import IsDOFArray

class IsVecDOFArray(Tag):
    pass


class IsFaceDOFArray(Tag):
    pass


class IsVecOpArray(Tag):
    pass

class ParameterValue(UniqueTag):
    
    def __init__(self, value):
        self.value = value
