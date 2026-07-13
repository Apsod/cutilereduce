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
                    return tuple(x.detach() for x in xs)
            else:
                mock = out.new_zeros(out.shape)
                mock.normal_()
                def bw(x):
                    (x * mock).sum().backward()
                    return (x.detach(),)
        out = bw(out)
        ret[k] = {}
        ret[k]['fwd'] = out
        grads = []
        for arg in args:
            if arg.grad is not None:
                grads.append(arg.grad.clone())
            ret[k]['bwd'] = tuple(grads)
    return ret
            


