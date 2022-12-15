# metaclass structure (helps with autoreload and isinstance)

from enum import Enum

class MetaFactor:
    pass

class MetaTerm:
    pass

class MetaFormula:
    pass

class MetaReal:
    pass

class MetaCateg:
    pass

class Drop(Enum):
    NONE = 0
    FIRST = 1
    VALUE = 2
