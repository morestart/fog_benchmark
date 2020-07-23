from numpy import array


def split_shift_data(step, seq_x):
    cache_x = list()
    for i in range(len(seq_x)):
        end_ix = i + step
        if end_ix > len(seq_x):
            break
        _x = seq_x[i: end_ix, :]

        cache_x.append(_x)

    return array(cache_x)
