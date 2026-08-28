from dataclasses import dataclass
from .stage_domain import StageDomain
from .stage_buffer import StageBufferBundle, WRITE, READ, RESIDENT

@dataclass(frozen=True)
class WorkModel:
    pass

@dataclass(frozen=True)
class KernelStage:
    name: str
    domain: StageDomain
    buffers: StageBufferBundle
    work: WorkModel

    @property
    def nonhiding_traffic(self):
        return (
            self.buffers.write.accessed_bytes + 
            self.buffers.read.persistent.accessed_bytes
        )

    @property
    def streamed_traffic(self):
        return self.buffers.read.streamed.accessed_bytes
    
