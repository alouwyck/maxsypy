import numpy as np
from scipy.special import exp1, erfc, k0, k1, i0, i1
from scipy.optimize import root
from math import factorial, log
import warnings


def thiem(r, T, Q, r_out, h_out=0.0):
    """
    Calculate hydraulic head at given distances r according to the Thiem formula for steady confined flow.

    Parameters
    ----------
    r : array_like
      Radial distances [L], which should be smaller than `r_out`.
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
        Pumping rate [L³/T] of the well.
    r_out : float
          Radial distance [L] of the outer aquifer boundary.
    h_out : float, default: 0.0
          Hydraulic head [L] at the outer aquifer boundary at distance `r_out`.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.
    """
    r = np.array(r)
    return h_out + Q / 2 / np.pi / T * np.log(r_out / r)


def dupuit(r, K, h0, Q, r_out):
    """
    Calculate hydraulic head at given distances r according to the Dupuit formula for steady unconfined flow.

    Parameters
    ----------
    r : array_like
      Radial distances [L], which should be smaller than `r_out`.
    K : float
      Aquifer conductivity [L/T].
    h0 : float
       Initial hydraulic head [L] which is the initial aquifer thickness before pumping.
       `h0` is also the constant head at the outer aquifer boundary at distance `r_out`.
    Q : float
      Pumping rate [L³/T] of the well (which is negative in case of extraction).
    r_out : float
          Radial distance [L] of the outer aquifer boundary.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.
    """
    return np.sqrt(h0**2 + Q / np.pi / K * np.log(r_out / r))


def island(r, T, Q, N, r_out, r_in=0.0, h_out=0.0):
    """
    Calculate the solution for steady flow to a pumping well in the center of an island with recharge.
    The solution is obtained by superimposing the Thiem formula and the equation for a circular infiltration area.

    Parameters
    ----------
    r : array_like
      Radial distances [L], which should be smaller than `r_out`.
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
      Pumping rate [L³/T].
    N : float
      Infiltration flux [L/T].
    r_out : float
          Radial distance [L] of the outer aquifer boundary.
    r_in : float, default: 0.0
         Pumping well radius [L], which coincides with the inner model boundary.
    h_out : float, default: 0.0
          Hydraulic head [L] at the outer aquifer boundary at distance `r_out`.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at given distances `r`. The shape of `h` is the same as the shape of `r`.
    """
    r = np.array(r)
    return h_out + (N * r_in**2 - Q / np.pi) / 2 / T * np.log(r / r_out) + N / 4 / T * (r_out**2 - r**2)


def deglee(r, T, Q, r_in=0.0, c_top=np.inf, h_top=0.0, c_bot=np.inf, h_bot=0.0):
    """
    Simulate steady flow to a pumping well in a leaky aquifer, which extracts water at a constant pumping rate.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    Q : float
      Pumping rate [L³/T] of the well.
    r_in : float, default: 0.O
         Pumping well radius [L], which coincides with the inner model boundary.
    c_top : float, default: inf
          Vertical resistance [T] of the aquitard overlying the aquifer.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: inf
          Vertical resistance [T] of the aquitard underlying the aquifer.
    h_bot : float, default: 0.0
          Constant head [L] of the lower boundary condition.

    Returns
    -------
    h : ndarray
      Hydraulic heads [L] at distances `r`.
      The length of `h` equals the length of `r`.
    """
    r = np.array(r)
    d = 1 / c_top / T + 1 / c_bot / T
    sd = np.sqrt(d)
    xi = r_in * sd
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
        ki = k1(xi) * xi
    if np.isnan(ki):
        ki = 1.0
    if np.isinf(c_top) and not np.isinf(c_bot):
        h1 = 0.0
        h2 = h_bot
    elif np.isinf(c_bot) and not np.isinf(c_top):
        h1 = h_top
        h2 = 0.0
    else:
        c_tot = c_top + c_bot
        h1 = h_top * c_bot / c_tot
        h2 = h_bot * c_top / c_tot
    return h1 + h2 + Q / 2 / np.pi / T * k0(r * sd) / ki


def theis(r, t, T, S, Q, h_out=0.0):
    """
    Simulate transient flow to a pumping well in a confined aquifer.
    The well has an infinitesimal radius and extracts water at a constant pumping rate.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-].
    Q : float
      Pumping rate [L³/T] of the well.
    h_out : float, default: 0.0
          Constant head [L] at the outer boundary condition, which is also the initial head in the aquifer.

    Returns
    -------
    h : ndarray
      Hydraulic head [L] at distances `r` and times `t`.
      Shape of `h` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.
    """
    t, r = np.meshgrid(t, r)
    return h_out + Q / 4 / np.pi / T * exp1(r * r * S / 4 / t / T)
    
    
def edelman(r, t, T, S, h_in=None, Q=None, h_out=0.0):
    """
    Simulate transient parallel flow in a semi-infinite aquifer.

    At the inner model boundary, a constant head `h_in` or a constant discharge `Q` is defined.
    This means either input parameter `h_in` or input parameter `Q` is assigned, but not both.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-].
    h_in : float
         Constant head [L] at the inner boundary condition.
    Q : float
      Constant discharge [L³/T/L] through the inner model boundary.
    h_out : float, default: 0.0
          Constant head [L] at the outer boundary condition, which is also the initial head in the aquifer.

    Returns
    -------
    h : ndarray
      Hydraulic head [L] at distances `r` and times `t`.
      Shape of `h` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.
    """
    t, r = np.meshgrid(t, r)
    u = r * np.sqrt(S / 4 / t / T)
    if Q is None:
        return h_out + (h_in - h_out) * erfc(u)
    else:
        return h_out + Q * (2 * np.sqrt(t / np.pi / T / S) * np.exp(-u**2) - r / T * erfc(u))


def hantush_jacob(r, t, T, S, Q, c_top, h_top=0.0, ns=12):
    """
    Simulate transient flow to a pumping well in a leaky aquifer.
    The well has an infinitesimal radius and extracts water at a constant pumping rate.

    The solution is obtained from numerical inversion of the exact analytical solution in Laplace space.
    There is also the option to apply a fast approximation of the Hantush Well function W.
    See input parameter `ns`.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    T : float
      Aquifer transmissivity [L²/T].
    S : float
      Aquifer storativity [-].
    Q : float
      Pumping rate [L³/T] of the well.
    c_top : float
          Vertical resistance [T] of the aquitard overlying the aquifer.
    h_top : float, default: 0.0
          Constant head [L] of the upper boundary condition, which is also the initial head in the aquifer.
    ns : int, default: 12
       Stehfest number - must be a positive, even integer.
       If `ns` is `None`, then a fast approximation of the Hantush Well function is applied.

    Returns
    -------
    h : ndarray
      Hydraulic head [L] at distances `r` and times `t`.
      Shape of `h` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.
    """
    if ns is None:  # Fast approximation of W
        t, r = np.meshgrid(t, r)
        rho = r / np.sqrt(T*c_top)
        tau = np.log(2*t/rho/c_top/S)
        return h_top + Q/4/np.pi/T * W_approx(rho, tau)
    else:  # Laplace
        r = np.array(r)
        if r.ndim == 0:
            r = r[np.newaxis]
        hp = lambda r, p: h_top / p + Q / 2 / np.pi / T / p * k0(r * np.sqrt(S * p / T + 1 / T / c_top))
        h = [stehfest(lambda p: hp(ri, p), t, ns) for ri in r]
        return np.array(h)


def W_approx(rho, tau):
    '''
    Fast approximation of the Hantush well function.

    Parameters
    ----------
    rho : array_like
        Two-dimensional array with dimensionless distances.
    tau : array_like
        Two-dimensional array with dimensionless times.

    Returns
    -------
    F : ndarray
      Fast approximation of Hantush well function
      `F` has the same shape as the input arrays
    '''
    w = (exp1(rho) - k0(rho)) / (exp1(rho) - exp1(rho/2))
    b = tau <= 0
    F = np.zeros(rho.shape)
    if np.any(b):
        wb, rhob, taub = w[b], rho[b], tau[b]
        F[b] = wb*exp1(rhob/2*np.exp(-taub)) - (wb-1)*exp1(rhob*np.cosh(taub))
    b = ~b
    if np.any(b):
        wb, rhob, taub = w[b], rho[b], tau[b]
        F[b] = 2*k0(rhob) - wb*exp1(rhob/2*np.exp(taub)) + (wb-1)*exp1(rhob*np.cosh(taub))
    return F


def ernst(r, T, c, N, Q):
    """
    Simulate steady flow to a pumping well in a phreatic aquifer subject to uniform recharge and drainage.
    The well is fully penetrating and extracts water at a constant pumping rate.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    c : float
      Drainage resistance [T].
    N : float
      Infiltration flux [L/T].
    Q : float
      Pumping rate [L³/T] of the well. 
      Negative as the model only simulates extractions.
    
    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r`.
      The length of `s` equals the length of `r`.
    """
    r = np.array(r)
    if r.ndim == 0: r = r[np.newaxis]
    s = np.zeros(r.shape)
    R = find_R_ernst(T, c, N, Q)
    b = r < R
    s[b] = s_prox_ernst(r[b], T, c, N, Q, R)
    b = ~b
    s[b] = s_dist_ernst(r[b], T, c, N, R)
    return s


def find_R_ernst(T, c, N, Q):
    """
    Finds boundary between proximal and distal zone in the Ernst model
    
    Parameters
    ----------
    T : float
      Aquifer transmissivity [L²/T].
    c : float
      Drainage resistance [T].
    N : float
      Infiltration flux [L/T].
    Q : float
      Pumping rate [L³/T] of the well.
      Negative as the model only simulates extractions.
    
    Returns
    -------
    R : float
      Distance [L] of boundary between proximal and distal zone.
    """
    QD = Q / np.pi / N / T / c  # dimensionless pumping rate
    L = np.sqrt(T*c)  # leakage factor
    func = lambda rd: (2 * k1(rd/L) / k0(rd/L) + rd/L) * rd/L + QD
    return root(func, 1).x[0]
    

def s_prox_ernst(r, T, c, N, Q, R):
    """
    Calculates drawdown in the proximal zone of the Ernst model.
    
    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    c : float
      Drainage resistance [T].
    N : float
      Infiltration flux [L/T].
    Q : float
      Pumping rate [L³/T] of the well.
      Negative as the model only simulates extractions.
    R : float
      Distance [L] of boundary between proximal and distal zone.
    
    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r`.
      The length of `s` equals the length of `r`.
    """
    return N * c + Q / 2 / np.pi / T * np.log(r/R) - N / 4 / T * (R**2 - r**2)


def s_dist_ernst(r, T, c, N, R):
    """
    Calculates drawdown in the distal zone of the Ernst model.
    
    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    T : float
      Aquifer transmissivity [L²/T].
    c : float
      Drainage resistance [T].
    N : float
      Infiltration flux [L/T].
    R : float
      Distance [L] of boundary between proximal and distal zone.
    
    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r`.
      The length of `s` equals the length of `r`.
    """
    L = np.sqrt(T * c)  # leakage factor
    return  N * c * k0(r/L) / k0(R/L)


def butler(r, t, R, T, S, Q, ns=12):
    """
    Simulate transient flow to a pumping well in a confined aquifer, which extracts water at a constant pumping rate.
    The pumping well has a finite-thickness skin.

    The function applies the Stehfest algorithm to numerically invert the Laplace transform.

    Parameters
    ----------
    r : array_like
      One-dimensional array with the radial distances [L].
    t : array_like
      One-dimensional array with the simulation times [T].
    R : float
       Radius [L] of well-skin (well-radius is zero)
    T : array_like
      Skin and aquifer transmissivities [L²/T], so T = [T_skin, T_aquifer]
    S : array_like
      Skin and aquifer storativities [-], so S = [S_skin, S_aquifer]
    Q : float
      Pumping rate [L³/T] of the well.
    ns : int, default: `12`
       Number of Stehfest parameters

    Returns
    -------
    s : ndarray
      Drawdown [L] at distances `r` and times `t`.
      The shape of `s` is `(nr, nt)`, with `nr` the length of `r`, and `nt` the length of `t`.
    """

    N = lambda p: np.sqrt(S[0] / T[0] * p)
    A = lambda p: np.sqrt(S[1] / T[1] * p)
    NR = lambda p: R * N(p)
    AR = lambda p: R * A(p)
    I0NR = lambda p: i0(NR(p))
    I1NR = lambda p: i1(NR(p))
    K0NR = lambda p: k0(NR(p))
    K1NR = lambda p: k1(NR(p))
    K0AR = lambda p: k0(AR(p))
    K1AR = lambda p: k1(AR(p))
    TTAN = lambda p: T[1] / T[0] * A(p) / N(p)
    QT = lambda p: Q / 2 / np.pi / T[0] / p
    denominator = lambda p: (TTAN(p) * I0NR(p) * K1AR(p) + I1NR(p) * K0AR(p))
    s1 = lambda p, r: QT(p) * (k0(N(p)*r) + (K1NR(p)*K0AR(p) - TTAN(p)*K0NR(p)*K1AR(p)) * i0(N(p)*r) / denominator(p))
    s2 = lambda p, r: QT(p) * (K0NR(p)*I1NR(p) + K1NR(p)*I0NR(p)) * k0(A(p)*r) / denominator(p)

    r = np.array(r)
    t = np.array(t)
    s = np.zeros((len(r), len(t)))
    for i in range(len(r)):
        for k in range(len(t)):
            if r[i] <= R:
                s[i, k] = stehfest(lambda p: s1(p, r[i]), t[k], ns)
            else:
                s[i, k] = stehfest(lambda p: s2(p, r[i]), t[k], ns)
    return s


def stehfest(F, t, ns=12):
    """
    Stehfest algorithm for numerical inversion of Laplace transforms.

    Parameters
    ----------
    F : callable
      Function that calculates the Laplace transform. It has frequency parameter `p` [1/T] as input
      and returns the Laplace-transform `F(p)`. Input parameter `p` is a one-dimensional numpy array,
      and the returned output is also a one-dimensional numpy array with the same length as `p`.
    t : array_like
      One-dimensional array with the real times `t` [T].
    ns : int, default: 12
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    f : ndarray
      One-dimensional array with the numerically inverted values `f(t)`. The length of `f` equals the length of `t`.
    """
    t = np.array(t)
    if t.ndim == 0:
        t = t[np.newaxis]
    nt = len(t)
    ns = int(ns)
    ln2t = log(2) / t
    W = stehfest_weights(ns)
    f = np.zeros(nt)
    for k in range(ns):
        p = ln2t * (k + 1)
        f += W[k] * F(p)
    return f * ln2t


def stehfest_weights(ns):
    """
    Calculate weights required for applying the Stehfest algorithm.

    Called by function `stehfest`.

    Parameters
    ----------
    ns : int
       Number of terms considered in the Stehfest algorithm applied for the inversion of the Laplace solution.
       Must be a positive, even integer.

    Returns
    -------
    W : ndarray
      One-dimensional array with weights, length of `W` is equal to `ns`.
    """
    fac = lambda x: float(factorial(x))
    ns2 = ns // 2
    W = np.zeros(ns)
    for j in range(1, ns + 1):
        m = min(j, ns2)
        k_0 = (j + 1) // 2
        for k in range(k_0, m + 1):
            W[j - 1] += k ** ns2 * fac(2 * k) / fac(ns2 - k) / fac(k) / fac(k - 1) / fac(j - k) / fac(2 * k - j)
        W[j - 1] *= (-1) ** (ns2 + j)
    return W