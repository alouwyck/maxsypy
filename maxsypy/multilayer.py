from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eig, inv
from scipy.special import i0, k0, k1, i1
from scipy.linalg import LinAlgWarning
from math import factorial, log
import warnings


class Steady:
    """
    Class to simulate steady two-dimensional radial or parallel flow in a multilayer aquifer system.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    Q : array_like, default: `None`
      Discharges [L³/T] at inner model boundary. The length of `Q` is equal to the number of layers.
    h_in : array_like, default: `None`
         Constant heads [L] at inner model boundary. The length of `h_in` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: `inf`
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    h_top : float, default: `0.0`
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: `inf`
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.
    h_bot : float, default: `0.0`
          Constant head [L] of the lower boundary condition.
    r_in : float, default: `0.0`
         Radial or horizontal distance [L] of the inner model boundary.
    r_out : float, default: `inf`
          Radial or horizontal distance [L] of the outer model boundary.
    h_out : array_like, default: `None`
          Constant head [L] at the outer model boundary for each layer.
          The length of `h_out` is equal to the number of layers.
          By default, the constant heads at the outer boundary are zero.
    N : array_like, default: `None`
      Recharge flux [L/T] for each layer.
      The length of `N` is equal to the number of layers.
      By default, the recharge in each layer is zero.
    axi : boolean, default: `True`
        Radial flow is simulated if `True`, parallel flow otherwise.

    Attributes
    ----------
    nl : int
       Number of layers
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    h(r) :
         Calculate hydraulic head h [L] at given distances r [L].
    qh(r) :
          Calculate radial or horizontal discharge Qh [L³/T] at given distances r [L].
    """
    def __init__(self, T, Q=None, h_in=None, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf,
                 h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None, N=None, axi=True):
        self.T = self._to_array(T)  # (nl, )
        self.nl = len(self.T)  # int
        self.Q = None if Q is None else self._to_array(Q)  # (nl, )
        self.h_in = None if h_in is None else self._to_array(h_in)  # (nl, )
        self.c = np.array([]) if self.nl == 1 else self._to_array(c)  # (nl-1, )
        self.c_top = c_top  # float
        self.h_top = h_top  # float
        self.c_bot = c_bot  # float
        self.h_bot = h_bot  # float
        self.confined = np.all(np.isinf([self.c_top, self.c_bot]))  # bool
        self.r_in = r_in  # float
        self.r_out = r_out  # float
        self.h_out = self._check_array(h_out)  # (nl, )
        self.N = self._check_array(N)  # (nl, )
        self.axi = axi  # bool
        self.no_warnings = True
        self._initialized = False  # becomes True the first time _out_() has been called

    def h(self, r):
        return self._out_('_h_', r)

    def qh(self, r):
        return self._out_('_qh_', r)

    def _ini_(self):
        self._Ab_()
        self._eig_()
        self._bc_()

    def _Ab_(self):
        c = np.hstack((self.c_top, self.c, self.c_bot))
        Tc0 = 1 / (self.T * c[:-1])  # (nl, )
        Tc1 = 1 / (self.T * c[1:])  # (nl, )
        self._idx = np.diag_indices(self.nl)
        irow, icol = self._idx
        self._A = np.zeros((self.nl, self.nl))  # (nl, nl)
        self._A[irow, icol] = Tc0 + Tc1
        self._A[irow[:-1], icol[:-1] + 1] = -Tc1[:-1]
        self._A[irow[:-1] + 1, icol[:-1]] = -Tc0[1:]
        self._b = self.N / self.T  # (nl, )
        self._b[0] += Tc0[0] * self.h_top
        self._b[-1] += Tc1[-1] * self.h_bot

    def _eig_(self):
        self._d, self._V = eig(self._A)  # (nl, ), (nl, nl)
        self._d = np.real(self._d)
        self._inz = np.arange(self.nl)
        if self.confined:
            self._iz = np.argmin(np.abs(self._d))
            self._inz = np.setdiff1d(self._inz, self._iz)
        if len(self._inz) > 0:
            self._sd = np.sqrt(self._d[self._inz])
        self._iV = inv(self._V)
        self._v = np.dot(self._iV, self._b)

    def _bc_(self):
        self._w = np.dot(self._iV, self.h_out)  # (nl, )
        self._alpha = np.zeros(self.nl)  # (nl, )
        self._beta = np.zeros(self.nl)  # (nl, )
        if self.axi:
            self._bessel_()
            if self.Q is not None:
                self._bc_axi_Q_()
            else:
                self._bc_axi_H_()
        else:
            if self.Q is not None:
                self._bc_par_Q_()
            else:
                self._bc_par_H_()

    def _bc_axi_Q_(self):
        self._q = np.dot(self._iV, self.Q / self.T / 2 / np.pi)  # (nl, )
        if self.confined:
            self._alpha[self._iz] = -self._q[self._iz] + self._v[self._iz] * self.r_in ** 2 / 2
            self._beta[self._iz] = self._w[self._iz] + self._v[self._iz] * self.r_out ** 2 / 4 - self._alpha[self._iz] * np.log(self.r_out)
        if len(self._inz) > 0:
            nominator = self._i1_in * self._k0_out + self._k1_in * self._i0_out
            wvd = self._w[self._inz] - self._v[self._inz] / self._d[self._inz]
            self._alpha[self._inz] = (wvd * self._k1_in - self._q[self._inz] * self._k0_out) / nominator
            self._beta[self._inz] = (wvd * self._i1_in + self._q[self._inz] * self._i0_out) / nominator
            i = np.isnan(self._beta[self._inz])
            if np.any(i):
                j = self._inz[i]
                self._beta[j] = self._q[j] / self._k1_in[i]  # if r_out -> inf

    def _bc_axi_H_(self):
        self._u = np.dot(self._iV, self.h_in)  # (nl, )
        if self.confined:
            r_in2, r_out2 = self.r_in ** 2, self.r_out ** 2
            ln_r_in, ln_r_out = np.log(self.r_in), np.log(self.r_out)
            u, w = self._u[self._iz], self._w[self._iz]
            v4 = self._v[self._iz] / 4
            nominator = ln_r_in - ln_r_out
            self._alpha[self._iz] = (u - w + v4 * (r_in2 - r_out2)) / nominator
            self._beta[self._iz] = (ln_r_in * (v4 * r_out2 + w) - ln_r_out * (v4 * r_in2 + u)) / nominator
        if len(self._inz) > 0:
            vd = self._v[self._inz] / self._d[self._inz]
            wvd, uvd = self._w[self._inz] - vd, self._u[self._inz] - vd
            nominator = self._i0_in * self._k0_out - self._k0_in * self._i0_out
            self._alpha[self._inz] = (uvd * self._k0_out - wvd * self._k0_in) / nominator
            self._beta[self._inz] = (wvd * self._i0_in - uvd * self._i0_out) / nominator
            i = np.isnan(self._beta[self._inz])
            if np.any(i):
                j = self._inz[i]
                self._beta[j] = uvd[j] / self._k0_in[i]  # if r_out -> inf

    def _bc_par_Q_(self):
        self._q = np.dot(self._iV, self.Q / self.T)  # (nl, )
        if self.confined:
            self._alpha[self._iz] = self._v[self._iz] * self.r_in - self._q[self._iz]
            self._beta[self._iz] = self._w[self._iz] + self._v[self._iz] * self.r_out ** 2 / 2 - self._alpha[self._iz] * self.r_out
        if len(self._inz) > 0:
            e_in, e_out = np.exp(self.r_in * self._sd), np.exp(self.r_out * self._sd)
            nominator = self._sd * (e_in / e_out + e_out / e_in)
            wvd = self._w[self._inz] - self._v[self._inz] / self._d[self._inz]
            self._alpha[self._inz] = (wvd * self._sd / e_in - self._q[self._inz] / e_out) / nominator
            self._beta[self._inz] = (wvd * self._sd * e_in + self._q[self._inz] * e_out) / nominator
            i = np.isnan(self._beta[self._inz])
            if np.any(i):
                j = self._inz[i]
                self._beta[j] = self._q[j] * e_in[i] / self._sd[i]  # if r_out -> inf

    def _bc_par_H_(self):
        self._u = np.dot(self._iV, self.h_in)  # (nl, )
        if self.confined:
            r_in2, r_out2 = self.r_in ** 2, self.r_out ** 2
            nominator = self.r_in - self.r_out
            u, w = self._u[self._iz], self._w[self._iz]
            v2 = self._v[self._iz] / 2
            self._alpha[self._iz] = (v2 * (r_in2 - r_out2) + u - w) / nominator
            self._beta[self._iz] = (self.r_in * (v2 * r_out2 + w) - self.r_out * (v2 * r_in2 + u)) / nominator
        if len(self._inz) > 0:
            e_in, e_out = np.exp(self.r_in * self._sd), np.exp(self.r_out * self._sd)
            nominator = e_in / e_out - e_out / e_in
            vd = self._v[self._inz] / self._d[self._inz]
            wvd, uvd = self._w[self._inz] - vd, self._u[self._inz] - vd
            self._alpha[self._inz] = (uvd / e_out - wvd / e_in) / nominator
            self._beta[self._inz] = (wvd * e_in - uvd * e_out) / nominator
            i = np.isnan(self._beta[self._inz])
            if np.any(i):
                j = self._inz[i]
                self._beta[j] = uvd[i] * e_in[i]  # if r_out -> inf

    def _out_(self, method, r):
        r = self._to_array(r)
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            if not self._initialized:
                self._ini_()
                self._initialized = True
            method = getattr(self, method)  # method _h_ or _qh_ (convert string to function)
            nr = len(r)
            out = np.zeros((self.nl, nr))  # (nl, nr)
            for i in range(nr):
                out[:, i] = method(r[i])
            return out

    def _h_(self, r):
        if self.axi:
            g = np.zeros(self.nl)  # (nl, )
            if self.confined:
                g[self._iz] = self._alpha[self._iz] * np.log(r) + self._beta[self._iz] - self._v[self._iz] * r ** 2 / 4
            if len(self._inz) > 0:
                x = self._sd * r
                i0x = i0(x)
                i0x[np.isnan(i0x)] = np.inf  # x -> inf: i0(x) = inf
                alpha_i0x = self._alpha[self._inz] * i0x
                alpha_i0x[np.isnan(alpha_i0x)] = 0.0  # if alpha = 0 and I0 = Inf
                g[self._inz] = alpha_i0x + self._beta[self._inz] * k0(x) + self._v[self._inz] / self._d[self._inz]
            return np.dot(self._V, g)
        else:
            g = np.zeros(self.nl)  # (nl, )
            if self.confined:
                g[self._iz] = self._alpha[self._iz] * r + self._beta[self._iz] - self._v[self._iz] * r ** 2 / 2
            if len(self._inz) > 0:
                ex = np.exp(self._sd * r)
                alpha_ex = self._alpha[self._inz] * ex
                alpha_ex[np.isnan(alpha_ex)] = 0.0  # if alpha = 0 and exp(x) = inf
                g[self._inz] = alpha_ex + self._beta[self._inz] / ex + self._v[self._inz] / self._d[self._inz]
            return np.dot(self._V, g)

    def _qh_(self, r):
        if self.axi:
            dg = np.zeros(self.nl)  # (nl, )
            if self.confined:
                dg[self._iz] = self._alpha[self._iz] - self._v[self._iz] * r ** 2 / 2
            if len(self._inz) > 0:
                x = self._sd * r
                xi1x = x * i1(x)
                xk1x = x * k1(x)
                xk1x[np.isnan(xk1x)] = 1.0  # r -> 0: x_in * k1(x_in) = 1
                alpha_xi1x = self._alpha[self._inz] * xi1x
                alpha_xi1x[np.isnan(alpha_xi1x)] = 0.0  # if alpha = 0 and I1 = Inf
                dg[self._inz] = alpha_xi1x - self._beta[self._inz] * xk1x
            return 2 * np.pi * self.T * np.dot(self._V, -dg)
        else:
            dg = np.zeros(self.nl)  # (nl, )
            if self.confined:
                dg[self._iz] = self._alpha[self._iz] - self._v[self._iz] * r
            if len(self._inz) > 0:
                ex = np.exp(self._sd * r)
                alpha_ex = self._alpha[self._inz] * ex
                alpha_ex[np.isnan(alpha_ex)] = 0.0  # if alpha = 0 and exp(x) = inf
                dg[self._inz] = self._sd * (alpha_ex + self._beta[self._inz] / ex)
            return self.T * np.dot(self._V, -dg)

    @staticmethod
    def _to_array(arr, dtype=float):
        arr = np.array(arr, dtype=dtype)
        if arr.ndim == 0:
            arr = arr[np.newaxis]
        return arr

    def _check_array(self, arr, dtype=float, n=None):
        if arr is None:
            return np.zeros(self.nl if n is None else n, dtype=dtype)
        else:
            return self._to_array(arr, dtype)

    def _bessel_(self):
        if len(self._inz) > 0:
            x_out = self._sd * self.r_out
            self._i0_out = i0(x_out)
            self._i0_out[np.isnan(self._i0_out)] = np.inf  # x_out -> inf: i0(x_out) = inf
            self._k0_out = k0(x_out)
            if self.Q is not None:
                x_in = self._sd * self.r_in
                self._i1_in = x_in * i1(x_in)
                self._k1_in = x_in * k1(x_in)
                self._k1_in[np.isnan(self._k1_in)] = 1.0  # r_in -> 0: x_in * k1(x_in) = 1
            else:
                x_in = self._sd * self.r_in
                self._i0_in = i0(x_in)
                self._k0_in = k0(x_in)
				

class Transient:
    """
    Class to simulate transient two-dimensional radial or parallel flow in a multilayer aquifer system.

    Parameters
    ----------
    T : array_like
      Layer transmissivities [L²/T]. The length of `T` is equal to the number of layers.
    S : array_like
      Layer storativities [-]. The length of `S` is equal to the number of layers.
    Q : array_like, default: `None`
      Discharges [L³/T] at inner model boundary. The length of `Q` is equal to the number of layers.
    h_in : array_like, default: `None`
         Constant heads [L] at inner model boundary. The length of `h_in` is equal to the number of layers.
    c : array_like
      Vertical resistances [T] between layers. The length of `c` is the number of layers minus one.
    c_top : float, default: `inf`
          Vertical resistance [T] of the upper boundary of the aquifer system.
          By default, the upper boundary is impervious.
    h_top : float, default: `0.0`
          Constant head [L] of the upper boundary condition.
    c_bot : float, default: `inf`
          Vertical resistance [T] of the lower boundary of the aquifer system.
          By default, the lower boundary is impervious.
    h_bot : float, default: `0.0`
          Constant head [L] of the lower boundary condition.
    r_in : float, default: `0.0`
         Radial or horizontal distance [L] of the inner model boundary.
    r_out : float, default: `inf`
          Radial or horizontal distance [L] of the outer model boundary.
    h_out : array_like, default: `None`
          Constant head [L] at the outer model boundary for each layer.
          The length of `h_out` is equal to the number of layers.
          By default, the constant heads at the outer boundary are zero.
    N : array_like, default: `None`
      Recharge flux [L/T] for each layer.
      The length of `N` is equal to the number of layers.
      By default, the recharge in each layer is zero.
    nstehfest: int, default: `10`
             Number of Stehfest parameters
    axi : boolean, default: `True`
        Radial flow is simulated if `True`, parallel flow otherwise.

    Attributes
    ----------
    nl : int
       Number of layers
    no_warnings : bool, default: True
                If `True`, the following warnings are suppressed: `RunTimeWarning` and SciPy `LinAlgWarning`.

    Methods
    -------
    h(r, t) :
            Calculate hydraulic head h [L] at given distances r [L] and times [T].
    qh(r, t) :
             Calculate radial or horizontal discharge Qh [L³/T] at given distances r [L] and times [T].
    """

    def __init__(self, T, S, Q=None, h_in=None, c=None, c_top=np.inf, h_top=0.0, c_bot=np.inf,
                 h_bot=0.0, r_in=0.0, r_out=np.inf, h_out=None, N=None, nstehfest=10, axi=True):
        self.T = self._to_array(T)  # (nl, )
        self.nl = len(self.T)  # int
        self.S = self._to_array(S)  # (nl, )
        self.nu = self.S / self.T  # (nl, )
        self.Q = None if Q is None else self._to_array(Q)  # (nl, )
        self.h_in = None if h_in is None else self._to_array(h_in)  # (nl, )
        self.c = np.array([]) if self.nl == 1 else self._to_array(c)  # (nl-1, )
        self.c_top = c_top  # float
        self.h_top = h_top  # float
        self.c_bot = c_bot  # float
        self.h_bot = h_bot  # float
        self.confined = np.all(np.isinf([self.c_top, self.c_bot]))  # bool
        self.r_in = r_in  # float
        self.r_out = r_out  # float
        self.h_out = self._check_array(h_out)  # (nl, )
        self.N = self._check_array(N)  # (nl, )
        self.nstehfest = int(nstehfest)  # int
        self.axi = axi  # bool
        self.no_warnings = True
        self._initialized = False  # becomes True the first time _out_() has been called
        self._steady = Steady(T=T, Q=Q, h_in=h_in, c=c, c_top=c_top, h_top=h_top, c_bot=c_bot,
                              h_bot=h_bot, r_in=r_in, r_out=r_out, h_out=h_out, N=N, axi=axi)

    def h(self, r, t):
        return self._out_('_h_', r, t)

    def qh(self, r, t):
        return self._out_('_qh_', r, t)

    def _ini_(self):
        self._stehfest_weights()
        self._steady.confined = False
        self._steady._initialized = True
        self._steady._Ab_()
        self._A = self._steady._A.copy()
        self._b = self._steady._b.copy()

    def _Ab_(self, p):
        self._steady._A[self._steady._idx] = self._A[self._steady._idx] + self.nu * p
        self._steady._b = self._b / p + self.nu * self.h_out
        self._steady.h_out = self.h_out / p
        if self.Q is not None:
            self._steady.Q = self.Q / p
        else:
            self._steady.h_in = self.h_in / p

    def _eig_(self):
        self._steady._eig_()

    def _bc_(self):
        self._steady._bc_()

    def _out_(self, method, r, t):
        r = self._to_array(r)
        t = self._to_array(t)
        with warnings.catch_warnings():
            if self.no_warnings:
                warnings.filterwarnings('ignore', category=LinAlgWarning)  # suppress scipy.linalg warnings
                warnings.filterwarnings('ignore', category=RuntimeWarning)  # suppress runtime warnings
            if not self._initialized:
                self._ini_()
                self._initialized = True
            method = getattr(self, method)  # method _h_ or _qh_ (convert string to function)
            nr, nt = len(r), len(t)
            ln2t = log(2) / t
            out = np.zeros((self.nl, nr, nt))
            for i in range(nt):
                for k in range(self.nstehfest):
                    p = ln2t[i] * (k + 1)
                    self._Ab_(p)
                    self._eig_()
                    self._bc_()
                    out[:, :, i] += self._W[k] * method(r)
                out[:, :, i] *= ln2t[i]
            return out

    def _h_(self, r):
        return self._steady.h(r)

    def _qh_(self, r):
        return self._steady.qh(r)

    def _stehfest_weights(self):
        fac = lambda x: float(factorial(x))
        ns2 = self.nstehfest // 2
        self._W = np.zeros(self.nstehfest)  # (nstehfest, )
        for j in range(1, self.nstehfest + 1):
            m = min(j, ns2)
            k_0 = (j + 1) // 2
            for k in range(k_0, m + 1):
                self._W[j - 1] += k ** ns2 * fac(2 * k) / fac(ns2 - k) / fac(k) / fac(k - 1) / fac(j - k) / fac(
                    2 * k - j)
            self._W[j - 1] *= (-1) ** (ns2 + j)

    @staticmethod
    def _to_array(arr, dtype=float):
        arr = np.array(arr, dtype=dtype)
        if arr.ndim == 0:
            arr = arr[np.newaxis]
        return arr

    def _check_array(self, arr, dtype=float, n=None):
        if arr is None:
            return np.zeros(self.nl if n is None else n, dtype=dtype)
        else:
            return self._to_array(arr, dtype)