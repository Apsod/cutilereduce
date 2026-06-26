import cuda.tile as ct

from cutilereduce.core import *

sizes = dict(
    l = 128,
    r = 1024,
    h = 8,
    g = 4,
    dq = 128,
    dv = 128,
)

a100 = {
        PEAK_FLOPS: 312e12,
        BANDWIDTH: 1.55e12,
        MAX_RESIDENCY: 64*1024,
        MIN_PARALLELISM: 108,
        MIN_MMA_EFFICIENCY: 0.5,
        }


a = Meta.make(
        input = dict(
            q = Buffer.make('l h g dq', ct.bfloat16),
            k = Buffer.make('r h dq', ct.bfloat16),
            v = Buffer.make('r h dv', ct.bfloat16)
            ),
        output = dict(
            m = Buffer.make('l h g', ct.float32),
            e = Buffer.make('l h g', ct.float32),
            v = Buffer.make('l h g dv', ct.float32),
            ),
        intermediate = [
            Buffer.make('l h g r', ct.float32),
            ],
        work = Work.make(
            forward=[
                'h : l g : r : dq', 
                'h : l g : dv : r'
                ],
            recompute=[
                'h : l g : r : dq',
                ]
            ),
        batch = Dims.parse('l h g'),
        fold = Dims.parse('r'),
    )

e = Estimator.make(
        a,
        sizes=sizes,
        symbols=a100,
        )

fwd_sweep = e.fwd.pareto_sweep()
bwd_sweep = e.bwd.pareto_sweep()

print(fwd_sweep.df)
print(bwd_sweep.df)
