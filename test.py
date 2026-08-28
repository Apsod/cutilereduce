from cutilereduce.core.buffer import buffer_spec, bundle_spec, BufferRole, Internal, Input, Output, GradStorage

import cuda.tile as ct

input = bundle_spec(
    Input,
    ctx=buffer_spec("b d", ct.bfloat16, req_grad=True, default=0),
    trg=buffer_spec("v d", ct.bfloat16, req_grad=True, default=0),
    targets=buffer_spec("b", ct.int32, req_grad=False, default=-100),
)

output = bundle_spec(
    Output,
    z = buffer_spec('b', ct.float32, default=float('-inf')),
    l = buffer_spec('b', ct.float32, default=0),
)

grad_storage = input.as_grad()

grad_accumulator = bundle_spec(
    Internal('grad_acc'),
    z = buffer_spec('b', ct.float32, default=float('-inf')),
    g_z = buffer_spec('b', ct.float32, default=0),
    g_l = buffer_spec('b', ct.float32, default=0),
)

execution = bundle_spec(
    Internal('execution'),
    m = buffer_spec('b', ct.float32, default=float('-inf')),
    e = buffer_spec('b', ct.float32, default=0),
    u = buffer_spec('b', ct.float32, default=0),
)

intermediate = bundle_spec(
    Internal('intermediate'),
    logits = buffer_spec('b v', ct.float32)
)

for x in input, output, grad_accumulator, execution, intermediate, grad_storage:
    print(x)
