import numpy as np
from scipy.special import i0, k0, i1, k1, factorial as fact


def KGS_no_skin(t, H0, rw, rc, B, b, d, confined, Kr, Kz, Ss,
                ns=12, maxerr=1e-6, miniter=10, maxiter=500, htol=1e-5):
    '''
    Simulate KGS model (without well-skin)
    - t (array_like) is time [T]
    - H0 (float) is initial head change [L] in well
    - rw (float) is well-screen radius [L]
    - rc (float) is well-casing radius [L]
    - B (float) is aquifer thickness [L]
    - b (float) is screen length [L]
    - d (float) is distance between screen top and aquifer top
    - confined (bool) indicates whether the aquifer is confined (True) or phreatic (False)
    - Kr (float) is the horizontal conductivity [L/T]
    - Kz (float) is the vertical conductivity [L/T]
    - Ss (float) is the specific storage [1/L]
    - ns (int) is the Stehfest number (default is 12)
    - maxerr (float) is the stop criterion, the maximum absolute head difference (default is 1e-6)
    - miniter (int) is the minimum number of iterations (default is 10)
    - maxiter (int) is the maximum number of iterations (default is 500)
    - htol (float) is the normalized head tolerance (default is 1e-5)
    returns the simulated head changes in the well for given times t, the number of iterations,
    and the convergence error.
    '''
    # dimensionless parameters
    beta = B / b
    zeta = d / b
    rw2 = rw * rw
    rc2 = rc * rc
    A = Kz / Kr
    a = b * b / rw2
    psi = A / a
    R = rw2 * Ss * b / rc2
    tau = t * b * Kr / rc2
    nt = len(t)
    phreatic = not confined
    pi2 = np.pi * np.pi
    o2 = pi2 / beta / beta

    # stehfest parameters
    ns2 = ns // 2
    ln2t = np.log(2) / tau  # len(ln2t) = nt

    # Laplace parameter p and stehfest weights w
    # p.shape = w.shape = (nt, ns)
    p = np.empty((nt, ns))
    w = np.empty(ns)
    for j in range(1, ns+1):
        p[:, j-1] = j * ln2t
        k = np.arange(np.floor((j + 1) / 2), min(j, ns2) + 1, dtype=int)
        w[j-1] = np.sum(k**ns2 * fact(2*k, exact=True) /
                        fact(ns2-k, exact=True) / fact(k, exact=True) /
                        fact(k-1, exact=True) / fact(j-k, exact=True) /
                        fact(2*k-j, exact=True)) * \
                 (-1)**(ns2 + j)
    w = np.tile(w, (nt, 1))

    # f1 and phi
    u = 4 * beta / pi2
    c1 = np.pi / 2 / beta
    if phreatic:
        f0 = 0
        u = 4 * u  # MUST BE 2 * u??
        c1 = c1 / 2
        o2 = o2 / 4
        sin_cos = np.sin
    else:
        nu0 = np.sqrt(R * p)
        f0 = k0(nu0) / k1(nu0) / nu0 / beta / 2  # f0.shape = (nt, ns)
        sin_cos = np.cos
    c2 = c1 * (1 + 2 * zeta)

    def f1(n):
        # n is scalar
        # f.shape = (nt, ns)
        n2 = n * n
        nu = np.sqrt(psi * o2 * n2 + R * p)
        f = k0(nu) / n2 / k1(nu) / nu * \
            np.sin(n * c1)**2 * sin_cos(n * c2)**2
        return f

    def phi(f):
        # f.shape = (nt, ns)
        omega = f0 + f * u
        f = omega / (1 + omega * p)
        return f

    # back transform to obtain head change
    f = f1(1)
    h = np.sum(w * phi(f), axis=1) * ln2t  # len(h) = nt
    err = np.Inf
    i = 1
    n = 2 + phreatic
    while (i < miniter or err > maxerr) and i < maxiter:
        df = f1(n)
        if np.nanmax(np.abs(df)) > 1e-16:  # f1 can be periodically equal to zero
            f = f + df  # size(f) = [nt, ns]
            hnew = np.sum(w * phi(f), axis=1) * ln2t  # len(hnew) = nt
            ok = ~np.isnan(hnew)
            err = max(np.abs(h[ok] - hnew[ok]))
            h[ok] = hnew[ok]
        i += 1
        n += 1 + phreatic

    # set normalized head smaller than given tolerance to zero
    # and denormalize
    h[h < htol] = 0
    h = h * H0

    return h, i, err


def KGS(t, H0, rw, rc, rs, B, b, d, confined, Kr, Kz, Ss, Krs, Kzs, Sss,
        ns=12, maxerr=1e-6, miniter=10, maxiter=500, htol=1e-5):
    '''
    Simulate KGS model (with finite-thickness skin)
    - t (array_like) is time [T]
    - H0 (float) is initial head change [L] in well
    - rw (float) is well-screen radius [L]
    - rc (float) is well-casing radius [L]
    - rs (float) is the outer radius [L] of the skin
    - B (float) is aquifer thickness [L]
    - b (float) is screen length [L]
    - d (float) is distance between screen top and aquifer top
    - confined (bool) indicates whether the aquifer is confined (True) or phreatic (False)
    - Kr (float) is the horizontal conductivity [L/T] of the aquifer
    - Kz (float) is the vertical conductivity [L/T] of the aquifer
    - Ss (float) is the specific storage [1/L] of the aquifer
    - Krs (float) is the horizontal conductivity [L/T] of the skin
    - Kzs (float) is the vertical conductivity [L/T] of the skin
    - Sss (float) is the specific storage [1/L] of the skin
    - ns (int) is the Stehfest number (default is 12)
    - maxerr (float) is the stop criterion, the maximum absolute head difference (default is 1e-6)
    - miniter (int) is the minimum number of iterations (default is 10)
    - maxiter (int) is the maximum number of iterations (default is 500)
    - htol (float) is the normalized head tolerance (default is 1e-5)
    returns the simulated head changes in the well for given times t, the number of iterations,
    and the convergence error.
    '''
    # dimensionless parameters
    rw2, rc2 = rw * rw, rc * rc
    epss = rs / rw
    beta, zeta = B / b, d / b
    A1, A2 = Kzs / Krs, Kz / Kr
    a, gamma = b**2 / rw2, Kr / Krs
    psi1, psi2 = A1 / a, A2 / a
    R1, R2 = gamma * rw2 * Sss * b / rc2, rw2 * Ss * b / rc2
    tau = t * b * Kr / rc2
    nt = len(t)
    phreatic = not confined
    pi2 = np.pi * np.pi
    o2 = pi2 / beta / beta

    # stehfest parameters
    ns2 = ns // 2
    ln2t = np.log(2) / tau  # len(ln2t) = nt

    # Laplace parameter p and stehfest weights w
    # p.shape = w.shape = (nt, ns)
    p = np.empty((nt, ns))
    w = np.empty(ns)
    for j in range(1, ns+1):
        p[:, j-1] = j * ln2t
        k = np.arange(np.floor((j + 1) / 2), min(j, ns2) + 1, dtype=int)
        w[j-1] = np.sum(k**ns2 * fact(2*k, exact=True) /
                        fact(ns2-k, exact=True) / fact(k, exact=True) /
                        fact(k-1, exact=True) / fact(j-k, exact=True) /
                        fact(2*k-j, exact=True)) * \
                 (-1)**(ns2 + j)
    w = np.tile(w, (nt, 1))

    # f1 and phi
    u = 4 * beta / pi2  # divide by 2 as omega is divided by 2
    c1 = np.pi / 2 / beta
    if phreatic:
        f0 = 0
        u = 4 * u  # 2 * u ??
        c1 = c1 / 2
        o2 = o2 / 4
        sin_cos = np.sin
    else:
        no1, no2 = np.sqrt(R1 * p), np.sqrt(R2 * p)  # nu1(n=0), nu2(n=0)
        x1, x2 = no1 * epss, no2 * epss
        Ng = no1 / no2 / gamma
        D1, D2 = k0(x1) * k1(x2) - Ng * k0(x2) * k1(x1), i0(x1) * k1(x2) + Ng * k0(x2) * i1(x1)
        f0 = (D2*k0(no1) - D1*i0(no1)) / (D2*k1(no1) + D1*i1(no1)) / no1 / beta / 2  # f0.shape = (nt, ns)
        sin_cos = np.cos
    c2 = c1 * (1 + 2 * zeta)

    def f1(n):
        # n is integer
        # f.shape = (nt, ns)
        n2 = n * n
        nu1, nu2 = np.sqrt(psi1 * o2 * n2 + R1 * p), np.sqrt(psi2 * o2 * n2 + R2 * p)
        x1, x2 = nu1 * epss, nu2 * epss
        Ng = nu1 / nu2 / gamma
        D1, D2 = k0(x1) * k1(x2) - Ng * k0(x2) * k1(x1), i0(x1) * k1(x2) + Ng * k0(x2) * i1(x1)
        f = (D2*k0(nu1) - D1*i0(nu1)) / (D2*k1(nu1) + D1*i1(nu1)) / nu1 / n2 * \
            np.sin(n * c1)**2 * sin_cos(n * c2)**2
        return f

    def phi(f):
        # f.shape = (nt, ns)
        omega = f0 + f * u
        go = gamma * omega
        f = go / (1 + go * p)
        return f

    # back transform to obtain head change
    f = f1(1)
    h = np.sum(w * phi(f), axis=1) * ln2t  # len(h) = nt
    err = np.Inf
    i = 1
    n = 2 + phreatic
    while (i < miniter or err > maxerr) and i < maxiter:
        df = f1(n)
        if np.nanmax(np.abs(df)) > 1e-16:  # f1 can be periodically equal to zero
            f = f + df  # size(f) = [nt, ns]
            hnew = np.sum(w * phi(f), axis=1) * ln2t  # len(hnew) = nt
            ok = ~np.isnan(hnew)
#            if np.any(~ok): print(np.sum(~ok))
            err = max(np.abs(h[ok] - hnew[ok]))
            h[ok] = hnew[ok]
        i += 1
        n += 1 + phreatic

    # set normalized head smaller than given tolerance to zero
    # and denormalize
    h[h < htol] = 0
    h = h * H0

    # output
    return h, i, err
