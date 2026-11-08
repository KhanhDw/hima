from .base_soma import BaseSoma
from .surrogate import SurrogateSpike, spike_fn
from .lif_soma import LIFSoma
from .adlif_soma import AdaptiveLIFSoma
from .elif_soma import ELIFSoma
from .izhikevich_soma import IzhikevichSoma
from .hh_soma import HodgkinHuxleySoma

# Backward-compatible name for projects that previously imported Soma.
Soma = LIFSoma

__all__ = [
    "BaseSoma",
    "Soma",
    "LIFSoma",
    "AdaptiveLIFSoma",
    "ELIFSoma",
    "IzhikevichSoma",
    "HodgkinHuxleySoma",
    "SurrogateSpike",
    "spike_fn",
]