import sys
from enum import Enum

if sys.version_info < (3, 10):
    from typing_extensions import TypeGuard
else:
    from typing import TypeGuard

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class OrderedEnum(Enum):
    def __ge__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value >= other.value

        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value > other.value

        return NotImplemented

    def __le__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value <= other.value

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if self.is_member_of_same_enum(other):
            return self.value < other.value

        return NotImplemented

    def is_member_of_same_enum(self: Self, other: object) -> TypeGuard[Self]:
        return self.__class__ is other.__class__


class HiddenValueEnum(Enum):
    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}>"
