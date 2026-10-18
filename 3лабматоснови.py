"""
Лабораторна робота №3 — «Швидке перетворення Фур'є»
Варіант n = 8. Вихідні дані — з ЛР №2 (N = 18 випадковий вектор; N = 8 відліків 01101000).
"""
import time, math, cmath, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



class Ops:
    def __init__(self): self.mul = 0; self.add = 0
    def reset(self): self.mul = self.add = 0
    # множення двох комплексних чисел: 4 дійсні множення + 2 додавання
    def cmul(self, a, b): self.mul += 4; self.add += 2; return a * b
    # додавання/віднімання комплексних: 2 дійсні додавання
    def cadd(self, a, b): self.add += 2; return a + b
    def csub(self, a, b): self.add += 2; return a - b


ops = Ops()


# ---------- ДПФ «в лоб» (для порівняння), O(N^2) ----------
def dft(x):
    N = len(x)
    W = [cmath.exp(-2j * math.pi * m / N) for m in range(N)]   # таблиця поворотних множників
    X = []
    for k in range(N):
        acc = ops.cmul(x[0], W[0])
        for n in range(1, N):
            acc = ops.cadd(acc, ops.cmul(x[n], W[(k * n) % N]))
        X.append(acc)
    return np.array(X)


# ---------- ШПФ radix-2, з проріджуванням за часом (ітеративне, in-place) ----------
def fft_radix2(x):
    x = np.asarray(x, dtype=complex).copy()
    N = len(x)
    if N & (N - 1):
        raise ValueError("N має бути степенем 2")
    # бітова інверсія індексів
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit; bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    size = 2
    while size <= N:
        half = size // 2
        wstep = cmath.exp(-2j * math.pi / size)
        for start in range(0, N, size):
            w = 1 + 0j
            for k in range(half):
                u = x[start + k]
                t = ops.cmul(w, x[start + k + half]) if k else x[start + k + half]
                x[start + k] = ops.cadd(u, t)
                x[start + k + half] = ops.csub(u, t)
                w *= wstep
        size *= 2
    return x


# ---------- ШПФ змішаної основи (Кулі–Тьюкі), довільне N = p1*p2*... ----------
def smallest_factor(N):
    for p in range(2, int(math.isqrt(N)) + 1):
        if N % p == 0:
            return p
    return N


def fft_mixed(x):
    x = list(x)
    N = len(x)
    p = smallest_factor(N)
    if p == N:                       # просте N -> пряме ДПФ
        return dft(x)
    m = N // p
    # p підпослідовностей x[r], x[r+p], ...
    subs = [fft_mixed(x[r::p]) for r in range(p)]
    X = [0j] * N
    for k in range(N):
        acc = subs[0][k % m]
        for r in range(1, p):
            w = cmath.exp(-2j * math.pi * r * k / N)
            acc = ops.cadd(acc, ops.cmul(w, subs[r][k % m]))
        X[k] = acc
    return np.array(X)


def measure(fn, x, repeat=50):
    ops.reset(); y = fn(x); m, a = ops.mul, ops.add
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(x)
    dt = (time.perf_counter() - t0) / repeat
    return y, dt, m, a


def main():
    # Вихідні дані з ЛР №2 (той самий генератор, що й у lab2.py)
    f18 = np.round(np.random.default_rng(2024).uniform(-5, 5, 18), 3)   # Частина I, N = 10 + 8
    s8 = np.array([int(c) for c in "0" + format(96 + 8, "b")], float)   # Частина II: 01101000
    rows = []

    # 1) N = 8 (Частина II ЛР2) — radix-2
    X_d, t_d, m_d, a_d = measure(dft, s8)
    X_f, t_f, m_f, a_f = measure(fft_radix2, s8)
    rows.append(("N=8, ДПФ", t_d, m_d, a_d)); rows.append(("N=8, ШПФ radix-2", t_f, m_f, a_f))
    print("N=8   max|ШПФ-ДПФ| =", np.max(np.abs(X_f - X_d)), " C_k =", np.round(X_f / 8, 6))

    # 2) N = 18 (Частина I ЛР2) — змішана основа 2·3·3
    Y_d, t_d, m_d, a_d = measure(dft, f18)
    Y_f, t_f, m_f, a_f = measure(fft_mixed, f18)
    rows.append(("N=18, ДПФ", t_d, m_d, a_d)); rows.append(("N=18, ШПФ 2·3·3", t_f, m_f, a_f))
    print("N=18  max|ШПФ-ДПФ| =", np.max(np.abs(Y_f - Y_d)),
          " max|ШПФ-numpy| =", np.max(np.abs(Y_f - np.fft.fft(f18))))

    # 3) N = 32: вектор ЛР2 доповнений нулями до степеня 2
    f32 = np.concatenate([f18, np.zeros(14)])
    Z_d, t_d, m_d, a_d = measure(dft, f32, 20)
    Z_f, t_f, m_f, a_f = measure(fft_radix2, f32, 20)
    rows.append(("N=32 (18+14 нулів), ДПФ", t_d, m_d, a_d))
    rows.append(("N=32 (18+14 нулів), ШПФ radix-2", t_f, m_f, a_f))
    print("N=32  max|ШПФ-ДПФ| =", np.max(np.abs(Z_f - Z_d)))

    for name, t, m, a in rows:
        print(f"{name:34s} t = {t*1e6:9.1f} мкс   mul = {m:6d}   add = {a:6d}")

    # 4) Залежність від N = 2^p
    Ns = [2 ** p for p in range(2, 11)]
    td, tf, md, mf, tnp = [], [], [], [], []
    rng = np.random.default_rng(1)
    for N in Ns:
        x = rng.standard_normal(N)
        rep = max(1, 2000 // N)
        _, t, m, _ = measure(dft, x, max(1, rep // 8)); td.append(t); md.append(m)
        _, t, m, _ = measure(fft_radix2, x, rep); tf.append(t); mf.append(m)
        t0 = time.perf_counter()
        for _ in range(200): np.fft.fft(x)
        tnp.append((time.perf_counter() - t0) / 200)
    scan = list(zip(Ns, td, tf, tnp, md, mf))
    print("\n   N     t_ДПФ, мс   t_ШПФ, мс   t_numpy, мс   mul_ДПФ   mul_ШПФ   виграш")
    for N, a, b, c, m1, m2 in scan:
        print(f"{N:5d} {a*1e3:11.3f} {b*1e3:11.3f} {c*1e3:12.4f} {m1:9d} {m2:9d} {m1/m2:8.1f}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].loglog(Ns, np.array(td) * 1e3, "o-", label="ДПФ (O(N²))")
    axs[0].loglog(Ns, np.array(tf) * 1e3, "s-", label="ШПФ radix-2 (O(N log N))")
    axs[0].loglog(Ns, np.array(tnp) * 1e3, "^-", label="numpy.fft")
    axs[0].set_xlabel("N"); axs[0].set_ylabel("час, мс"); axs[0].set_title("Час обчислення")
    axs[1].loglog(Ns, md, "o-", label="ДПФ: 4N²")
    axs[1].loglog(Ns, mf, "s-", label="ШПФ radix-2: O(N·log₂N)")
    axs[1].set_xlabel("N"); axs[1].set_ylabel("к-сть дійсних множень")
    axs[1].set_title("Кількість множень")
    for ax in axs: ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("compare.png", dpi=150); plt.close(fig)

    # спектр N=18, обчислений ШПФ
    k = np.arange(18)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].stem(k, np.abs(Y_f) / 18); axs[0].set_title("|C_k| (ШПФ), N = 18")
    axs[1].stem(k, np.angle(Y_f)); axs[1].set_title("arg C_k (ШПФ), N = 18")
    for ax in axs: ax.grid(alpha=.3); ax.set_xlabel("k")
    fig.tight_layout(); fig.savefig("fft_spectrum18.png", dpi=150); plt.close(fig)
    return rows, scan


if __name__ == "__main__":
    main()
