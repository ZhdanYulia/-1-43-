"""
Лабораторна робота №4 — «Інтегральне перетворення Фур'є»
Варіант n = 8 -> випадок a): f(t) = t^(2n) = t^16,  межі інтегрування [-N, N], N = 100*n = 800
w_k = 2*pi*k/T,  T = 4, 8, 16, 32, 64, 128
"""
import time
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n = 8
P = 2 * n            # степінь: f(t) = t^16
NLIM = 100 * n       # 800
TS = [4, 8, 16, 32, 64, 128]
K = 40               # k = 0..K
mp.mp.dps = 80       # підвищена точність (значення ~1e48, сильне взаємне знищення)


def f(t):
    return t ** P


# 1. k-й член інтегралу Фур'є: F(w_k) = ∫_{-N}^{N} f(t)[cos(-w_k t) + j sin(-w_k t)] dt
#    Обчислюється аналітично (інтегрування частинами), у довгій арифметиці:
#    ∫ t^m e^{-jwt} dt = -e^{-jwt} * Σ_{i=0}^{m} m!/(m-i)! t^(m-i) / (jw)^(i+1)
def fourier_integral_k(k, T, N=NLIM, m=P):
    w = 2 * mp.pi * k / T
    if k == 0:
        val = (mp.mpf(N) ** (m + 1) - (-mp.mpf(N)) ** (m + 1)) / (m + 1)
        return mp.mpc(val, 0)
    jw = mp.mpc(0, 1) * w

    def antideriv(t):
        t = mp.mpf(t)
        s = mp.mpc(0)
        fact = mp.mpf(1)
        for i in range(m + 1):
            s += fact * t ** (m - i) / jw ** (i + 1)
            fact *= (m - i)
        return -mp.exp(-jw * t) * s

    return antideriv(N) - antideriv(-N)


# Той самий k-й член — чисельно (формула Сімпсона), для перевірки
def fourier_integral_k_numeric(k, T, N=NLIM, M=2_000_000):
    w = 2 * np.pi * k / T
    t = np.linspace(-N, N, M + 1)
    h = 2 * N / M
    wts = np.ones(M + 1); wts[1:-1:2] = 4; wts[2:-1:2] = 2
    ft = f(t)
    re = h / 3 * np.sum(wts * ft * np.cos(-w * t))
    im = h / 3 * np.sum(wts * ft * np.sin(-w * t))
    return re + 1j * im


# 2. Спектр амплітуд |F(w_k)| = sqrt(Re^2 + Im^2)
def amplitude(F):
    return mp.sqrt(mp.re(F) ** 2 + mp.im(F) ** 2)


def spectrum(T, K=K):
    w, re, im, amp = [], [], [], []
    for k in range(K + 1):
        F = fourier_integral_k(k, T)
        w.append(float(2 * mp.pi * k / T)); re.append(float(mp.re(F)))
        im.append(float(mp.im(F))); amp.append(float(amplitude(F)))
    return np.array(w), np.array(re), np.array(im), np.array(amp)


# 3. Графіки Re F(w_k) та |F(w_k)| для T = 4..128
def main():
    t0 = time.perf_counter()
    res = {T: spectrum(T) for T in TS}
    print(f"Час обчислення {len(TS)}×{K+1} значень: {time.perf_counter()-t0:.2f} c")

    for T in TS:
        w, re, im, amp = res[T]
        print(f"\nT = {T}:  max|Im F| = {np.max(np.abs(im)):.2e}")
        for k in (0, 1, 2, 3, 5, 10, 20, 40):
            print(f"  k={k:2d} w={w[k]:8.4f}  Re F = {re[k]: .6e}  |F| = {amp[k]:.6e}")

    # перевірка аналітичного значення чисельним інтегруванням
    print("\nПеревірка (Сімпсон, 2e6 вузлів):")
    for T, k in [(4, 1), (16, 3), (128, 5), (128, 1)]:
        Fa = complex(fourier_integral_k(k, T)); Fn = fourier_integral_k_numeric(k, T)
        print(f"  T={T:3d} k={k}: аналіт. {Fa.real: .8e}  чисел. {Fn.real: .8e}  "
              f"відн. різниця {abs(Fa.real-Fn.real)/abs(Fa.real):.1e}")

    fig, axs = plt.subplots(3, 2, figsize=(12, 11))
    for ax, T in zip(axs.flat, TS):
        w, re, im, amp = res[T]
        ax.stem(w, re, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="Re F(w_k)")
        ax.plot(w, amp, "C3.--", lw=1, label="|F(w_k)|")
        ax.set_yscale("symlog", linthresh=1e38)
        ax.set_title(f"T = {T}"); ax.set_xlabel("w_k, рад/с"); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"$f(t)=t^{16}$, $[-800, 800]$: Re F(w_k) та |F(w_k)|, k = 0..40 (symlog)")
    fig.tight_layout(); fig.savefig("spectra_all.png", dpi=140); plt.close(fig)

    # окремі графіки для кожного T (лінійна шкала, нормовано на F(0))
    for T in TS:
        w, re, im, amp = res[T]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        axs[0].stem(w, re / re[0]); axs[0].set_title(f"Re F(w_k) / F(0),  T = {T}")
        axs[1].stem(w, amp / amp[0]); axs[1].set_title(f"|F(w_k)| / F(0),  T = {T}")
        axs[1].set_yscale("log")
        for ax in axs: ax.set_xlabel("w_k, рад/с"); ax.grid(alpha=.3, which="both")
        fig.tight_layout(); fig.savefig(f"spectrum_T{T}.png", dpi=140); plt.close(fig)
    return res


if __name__ == "__main__":
    main()
