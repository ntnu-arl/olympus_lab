from enum import IntEnum, auto


class InitializationScheme(IntEnum):
    DEFAULT = 0
    STANDING = auto()
    TOUCHDOWN = auto()
    INFLIGHT = auto()
    LANDED = auto()
    NUM_SCHEMES = auto()



