from dataclasses import dataclass
from .stage_domain import StageDomain
from .stage_buffer import StageBufferBundle
from .utilities import flat_tuple

@dataclass(frozen=True)
class WorkModel:
    pass

@dataclass(frozen=True)
class KernelStage:
    name: str
    domain: StageDomain
    bundle: StageBufferBundle
    work: WorkModel

    @property
    def read_buffers(self) -> Tuple[StageBuffer]:
        return flat_tuple(b for b in self.bundles if b.is_read)

    @property
    def write_buffers(self) -> Tuple[StageBuffer]:
        return flat_tuple(b for b in self.bundles if b.is_write)

    @property
    def resident_buffers(self) -> tuple[StageBuffer]:
        return flat_tuple(b for b in self.bundles if b.is_resident)

    @property
    def residency_partition(self): -> tuple[tuple[StageBuffer], tuple[StageBuffer]]:
        streamed = []
        resident = []
        for b in self.bundles:
            _streamed, _resident = b.residency_partition
            streamed = streamed + _streamed
            resident = resident + _resident
        return tuple(streamed), tuple(resident)

    @property
    def read_traffic(self):
        return sum(b.accessed_bytes for b in self.read_buffers)

    @property
    def write_buffers(self):
        return sum(b.accessed_bytes for b in self.write_buffers)



