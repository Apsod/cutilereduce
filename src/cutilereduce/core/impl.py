import cuda.tile as ct



def realize(grid, input, output, config, map_reduce, binary_combine):
    @ct.kernel
    def cutilereduce_kernel(*buffers):
        in_bufs = ct.static_eval(buffers[:len(input)])
        out_bufs = ct.static_eval(buffers[len(input):])
        bid = ct.bid(0)


