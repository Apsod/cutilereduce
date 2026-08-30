from sympy import Function, Piecewise, Symbol

READ = Symbol("READ")
WRITE = Symbol("WRITE")
GROUPS = Symbol("GROUPS")

PEAK_TENSOR_FLOPS = Symbol("PEAK_TENSOR_FLOPS")
PEAK_FLOPS = PEAK_TENSOR_FLOPS
BANDWIDTH = Symbol("BANDWIDTH")
SM_COUNT = Symbol("SM_COUNT")
MAX_PROGRAMS_PER_SM = Symbol("MAX_PROGRAMS_PER_SM")
SMEM_PER_SM = Symbol("SMEM_PER_SM")
ATOMIC_ADD = Symbol("ATOMIC_ADD")

FWD_CONTENTION = Function("FWD_CONTENTION")
BWD_CONTENTION = Function("BWD_CONTENTION")


def normal_write(residual_multiplicity, active_multiplicity):
    return 1


def atomic_add_write(residual_multiplicity, active_multiplicity):
    return Piecewise(
        (1, residual_multiplicity <= 1),
        (ATOMIC_ADD + BWD_CONTENTION(active_multiplicity), True),
    )

__all__ = [
    "READ",
    "WRITE",
    "GROUPS",
    "PEAK_FLOPS",
    "PEAK_TENSOR_FLOPS",
    "BANDWIDTH",
    "SM_COUNT",
    "MAX_PROGRAMS_PER_SM",
    "SMEM_PER_SM",
    "ATOMIC_ADD",
    "FWD_CONTENTION",
    "BWD_CONTENTION",
    "normal_write",
    "atomic_add_write",
]
