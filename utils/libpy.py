def n_even_chunks(l, n):
    """Yield n as even chunks as possible from l."""
    last = 0
    for i in range(1, n+1):
        cur = int(round(i * (len(l) / n)))
        yield l[last:cur]
        last = cur
def n_even_chunks_naive(l, n):
    ret=[];
    last = 0
    for i in range(1, n+1):
        cur = int(round(i * (len(l) / n)))
        ret.append( l[last:cur])
        last = cur
    return ret;