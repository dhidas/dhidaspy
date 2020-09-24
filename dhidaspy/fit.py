
def fwhm (x, y, ip=None):
    """
    Find the full width half max given x and y arrays.  Optionally specify a point ip of a local max

    x : list/array
    y : list/array
    ip : index of point about which to find fwhm (better be a max)
    """

    if ip is None:
        ip = y.index(max(y))

    halfmax = y[ip] / 2.

    for i in range(ip, 0, -1):
        if y[i] < halfmax:
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            dy = (halfmax - y[i]) / (y[i + 1] - y[i])
            x1 = x[i + 1] - dy / slope
            break
    for i in range(ip, len(x), 1):
        if y[i] < halfmax:
            slope = (y[i - 1] - y[i]) / (x[i - 1] - x[i])
            dy = (halfmax - y[i]) / (y[i - 1] - y[i])
            x2 = x[i - 1] - dy / slope
            break

    return x2 - x1
