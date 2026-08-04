from enum import Enum

class PeriodUnit(str, Enum):
    # https://build.fhir.org/datatypes.html#Timing
    s = "s"
    min = "min"
    h = "h"
    d = "d"
    wk = "wk"
    mo = "mo"
    a = "a"