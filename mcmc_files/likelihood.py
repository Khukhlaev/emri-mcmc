# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
# Packages required for EMRI waveforms
from few.waveform import GenerateEMRIWaveform

import h5py
from ldc.lisa.orbits import OrbitsFromFile
from ldc.lisa.projection import ProjectedStrain

from ldc.common.tools import window

from ldc.common.series import TDI, XYZ2AET
from ldc.lisa.noise import get_noise_model
from ldc.waveform.waveform import NumericHpHc


# Helping functions

def inner_product(lhsA, lhsE, rhsA, rhsE, SA, df):
    return 4.0 * df * np.sum(np.real(lhsA * np.conj(rhsA) + lhsE * np.conj(rhsE)) / SA)


def fourier(data, dt, n=0):
    """
    params: data - list like with elements - arrays-like of equal size
    return: list - fourier transforms, frequencies
    """
    if n == 0:
        n = data[0].size

    for i in range(len(data)):
        data[i] = np.fft.rfft(data[i], n)[1:]

    freq = np.fft.rfftfreq(n, d=dt)[1:]  # cause we want freq[0] != 0
    return data, freq


def crop_data(data, freq, fmin, fmax):
    """
    params: data - list like with elements - arrays-like of equal size; freq - array-like of original frequencies;
    return: list - cropped data, cropped frequencies
    """
    if fmin == 0 and fmax == np.inf:
        return data, freq

    n = freq.size
    imin, imax = 0, n - 1
    for i in range(n):
        if freq[i] > fmin:
            imin = i
            break
    for i in range(n - 1, -1, -1):
        if freq[i] < fmax:
            imax = i
            break
    for i in range(len(data)):
        data[i] = data[i][imin:imax]
    freq = freq[imin:imax]
    return data, freq


class LikelihoodCalculator:

    def __init__(self, Phi_phi0, Phi_theta0, Phi_r0, T, dt, priors):
        self.Phi_phi0 = Phi_phi0
        self.Phi_theta0 = Phi_theta0
        self.Phi_r0 = Phi_r0
        self.T = T  # in years
        self.dt = dt  # in seconds
        self.orbit_file = 'equalarmlength-orbits.h5'
        self.priors = priors

        with h5py.File(self.orbit_file) as f:
            L = f.attrs['L']

        self.orbits = OrbitsFromFile({
            'orbit_type': 'file',
            'filename': self.orbit_file,
            'nominal_arm_length': L},
            read_t0=False,
        )

        self.dA = None  # in freq. domain
        self.dE = None  # in freq. domain
        self.freq = None
        # self.addition = None  # equal to -1/2 (d, d)
        self.SA = None
        self.n = None  # size of original signal in time
        self.df = None

    def get_tdi(self, M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK):
        # Generating EMRI waveform
        gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform", use_gpu=False)
        h_aak = gen_wave(
            M,
            mu,
            a,
            p0,
            e0,
            x0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            self.Phi_phi0,
            self.Phi_theta0,
            self.Phi_r0,
            T=self.T,
            dt=self.dt,
        )

        end_index, = h_aak.shape
        t_merge = end_index * self.dt

        # Calculating response
        t_min = 0
        t_max = t_merge + 1000
        t = np.arange(t_min, t_max, self.dt)
        h_aak = np.concatenate([h_aak, np.zeros(t.size - h_aak.size)])

        hphc_num = NumericHpHc(t, h_aak.real, h_aak.imag, qS, phiS)
        projected_strain = ProjectedStrain(self.orbits)
        projected_strain.arm_response(t_min + 500, t_max - 500, self.dt, [hphc_num])

        # Calculating TDI
        X_ldc = projected_strain.compute_tdi_x(t)
        Y_ldc = projected_strain.compute_tdi_y(t)
        Z_ldc = projected_strain.compute_tdi_z(t)

        # Applying window function
        window_function = window(t)
        X_ldc *= window_function
        Y_ldc *= window_function
        Z_ldc *= window_function

        # Transform to A, E
        A, E, _ = XYZ2AET(X_ldc, Y_ldc, Z_ldc)

        return A, E

    def setup(self, M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK):
        A, E = self.get_tdi(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK)

        self.n = A.size
        [self.dA, self.dE], self.freq = fourier([A, E], self.dt)
        [self.dA, self.dE], self.freq = crop_data([self.dA, self.dE], self.freq, 6e-4, 2e-2)

        self.df = self.freq[1] - self.freq[0]

        # Setup noise model
        noise = get_noise_model("SciRDv1", self.freq, wd=1)
        self.SA = noise.psd(self.freq)
        # self.addition = - 0.5 * inner_product(self.dA, self.dE, self.dA, self.dE, self.SA, self.df)

        SN2 = 4.0 * self.df * np.sum((np.abs(self.dA) ** 2 + np.abs(self.dE) ** 2) / self.SA)
        print("Setup successful!")
        print("SNR of the original signal =", round(np.sqrt(SN2), 3))
        # plt.loglog(self.freq, np.abs(self.dA))
        # plt.loglog(self.freq, np.sqrt(self.SA))
        # plt.show()

    def setup_with_sangria(self, M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK):
        A, E = self.get_tdi(M, mu, a, p0, e0, x0, dist, qS, phiS, qK, phiK)

        sangria_fn = "LDC2_sangria_training_v2.h5"
        tdi_ts = TDI.load(sangria_fn, name="obs/tdi")
        tdi_mbhb_ts = TDI.load(sangria_fn, name="sky/mbhb/tdi")

        tdi_ts -= tdi_mbhb_ts
        tdi_ts.XYZ2AET()

        self.n = tdi_ts['A'].values.size
        A = np.concatenate([A, np.zeros(self.n - A.size)])
        E = np.concatenate([E, np.zeros(self.n - E.size)])
        tdi_ts['A'].values += A
        tdi_ts['E'].values += E

        [self.dA, self.dE], self.freq = fourier([tdi_ts['A'].values, tdi_ts['E'].values], self.dt)
        [self.dA, self.dE], self.freq = crop_data([self.dA, self.dE], self.freq, 5e-4, 2e-2)

        self.df = self.freq[1] - self.freq[0]

        # Setup noise model
        noise = get_noise_model("sangria", self.freq, wd=1)
        self.SA = noise.psd(self.freq)
        # self.addition = - 0.5 * inner_product(self.dA, self.dE, self.dA, self.dE, self.SA, self.df)

        # SN2 = 4.0 * self.df * np.sum((np.abs(self.dA) ** 2 + np.abs(self.dE) ** 2) / self.SA)
        print("Setup successful!")
        # print("SNR of the original signal =", round(np.sqrt(SN2), 3))
        # plt.loglog(self.freq, np.abs(self.dA), label="dA")
        # plt.loglog(self.freq, np.sqrt(self.SA), label="SA")
        # plt.xlabel("freq. [Hz]")
        # plt.legend()
        # plt.show()

    def loglikelihood(self, point, i=0, T=1):
        start = time.time()
        # assert len(point) == 11

        for n in range(len(self.priors)):
            if point[n] < self.priors[n][0] or point[n] > self.priors[n][1]:
                return -100000

        qS = np.arccos(point[7])
        qK = np.arccos(point[9])
        try:
            A, E = self.get_tdi(point[0], point[1], point[2], point[3], point[4], point[5], point[6], qS, point[8], qK,
                                point[10])
        except ValueError:
            return -100000

        [hA, hE], freq = fourier([A, E], self.dt, self.n)
        [hA, hE], freq = crop_data([hA, hE], freq, 5e-4, 2e-2)
        result = inner_product(self.dA, self.dE, hA, hE, self.SA, self.df) \
                 - 0.5 * inner_product(hA, hE, hA, hE, self.SA, self.df)
        end = time.time()
        print("Time for computing likelihood =", round(end - start, 2), "seconds, T = " + str(T) + ", i =", i)

        return result
