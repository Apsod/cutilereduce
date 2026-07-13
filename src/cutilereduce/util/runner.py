import torch

def run_grad(*args, **functions):
    ret = {}
    mock = None
    for k, f in functions.items():
        for arg in args:
            if arg.grad is not None:
                arg.grad.zero_()
        out = f(*args)
        if mock is None:
            if isinstance(out, tuple):
                mock = [o.new_zeros(o.shape) for o in out]
                for m in mock:
                    m.normal_()
                def bw(xs):
                    sum([(x * m).sum() for x, m in zip(xs, mock)]).backward()
            else:
                mock = out.new_zeros(out.shape)
                mock.normal_()
                def bw(x):
                    (x * mock).sum().backward()
        bw(out)
        ret[k] = []

        for arg in args:
            if arg.grad is not None:
                ret[k].append(arg.grad.clone())
    return ret
            


