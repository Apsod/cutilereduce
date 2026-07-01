import cuda.tile as ct
import polars as pl

from cutilereduce.core import *


a100 = {
        PEAK_FLOPS: 312e12,
        BANDWIDTH: 1.55e12,
        MIN_MMA_EFFICIENCY: 0.5,
        SM_COUNT: 108,
        MAX_PROGRAMS_PER_SM: 32,
        SMEM_PER_SM: 164*1024,
        }

l40s = {
        PEAK_FLOPS: 362e12,
        BANDWIDTH: 864e9,
        MIN_MMA_EFFICIENCY: 0.5,
        SM_COUNT: 142,
        MAX_PROGRAMS_PER_SM: 32,
        SMEM_PER_SM: 100*1024,
        }


#attention = Meta.make(
#        input = dict(
#            q = Buffer.make('l h g dq', ct.bfloat16),
#            k = Buffer.make('r h dq', ct.bfloat16),
#            v = Buffer.make('r h dv', ct.bfloat16)
#            ),
#        output = dict(
#            m = Buffer.make('l h g', ct.float32),
#            e = Buffer.make('l h g', ct.float32),
#            v = Buffer.make('l h g dv', ct.float32),
#            ),
#        intermediate = [
#            Buffer.make('l h g r', ct.float32),
#            ],
#        work = Work.make(
#            forward=[
#                'h : l g : r : dq', 
#                'h : l g : dv : r'
#                ],
#            recompute=[
#                'h : l g : r : dq',
#                ]
#            ),
#        batch = Dims.parse('l h g'),
#        fold = Dims.parse('r'),
#    )
#
#sizes = dict(
#    l = 128,
#    r = 1024,
#    h = 8,
#    g = 4,
#    dq = 128,
#    dv = 128,
#)

spec=dict(
        input = dict(
            ctx = Buffer.make('b d', ct.bfloat16, req_grad=True),
            trg = Buffer.make('v d', ct.bfloat16, req_grad=True),
            targets = Buffer.make('b', ct.int32),
        ),
        output = dict(
            m = Buffer.make('b', ct.float32),
            e = Buffer.make('b', ct.float32),
            u = Buffer.make('b', ct.float32),
        ),
        intermediate = [
            Buffer.make('b v', ct.float32),
            ],
        work = Work.make(
            forward=[
                ': b : v : d',
                ],
            recompute=[
                ': b : v : d',
                ],
            ),
        batch = Dims.parse('b'),
        fold = Dims.parse('v'),
    )

xentropy = Spec.make(
        **spec,
        )

sizes = dict(
        b = 1024*256,
        v = 1024*256,
        d = 128,
        )

e = Estimator.make(
        xentropy,
        sizes=sizes,
        symbols=a100,
        )

sweeper = Sweep(
        attributes = [
            'estimated_time', 'effective_traffic', 'effective_total_work', 'mma_efficiency', 'traffic',
            'contention', 'resident_programs_per_sm', 'group_size', 'effective_intensity_ratio',
            'residency_bytes', 'arithmetic_intensity', 'smem_utilization',
            ],
        filters = [
            pl.col('resident_programs_per_sm') >= 1,
            pl.col('group_size') >= 1,
            pl.col('mma_efficiency') == 1
            ],
        paretos = [
            'effective_total_work', 'effective_traffic'
            ]
        )

cols = 'estimated_time',

res = sweeper.run_all(e).select(pl.selectors.starts_with('cfg:'), *cols)
for phase, df in res.group_by('cfg:phase'):
    print(df.sort('estimated_time'))
