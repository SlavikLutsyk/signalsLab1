import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Функція f(x) = n * sin(π * n * x) на інтервалі [0, π]
def F(x, n=1):
    return n * np.sin(np.pi * n * x)

def A_k(k, n=1):
    return 0

def B_k(k, n=1):
    # Обчислення коефіцієнту b_k ряду Фур'є на інтервалі [0, π]
    integrand = lambda x: F(x, n) * np.sin(k * x)
    result, _ = quad(integrand, 0, np.pi)
    return (2 / np.pi) * result

def FourierSeries(x, N, n=1):
    series_sum = 0
    for k in range(1, N+1):
        series_sum += B_k(k, n) * np.sin(k * x)
    return series_sum

def plot_harmonics(aCoeffs, bCoeffs, N):
    # Plot a_k (всі a_k = 0)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.stem(range(len(aCoeffs)), aCoeffs, basefmt=" ")
    plt.title('Коефіцієнти a_k')
    plt.xlabel('k')
    plt.ylabel('a_k')

    # Plot b_k
    plt.subplot(1, 2, 2)
    plt.stem(range(1, N+1), bCoeffs, basefmt=" ")
    plt.title('Коефіцієнти b_k')
    plt.xlabel('k')
    plt.ylabel('b_k')


    plt.tight_layout()
    plt.show()


def plot_function_and_approximation(xValues, N):
    # Початкова функція
    initial_function = np.array([F(x) for x in xValues])
    # Наближення рядом Фур'є
    approx_function = np.array([FourierSeries(x, N) for x in xValues])


    plt.figure(figsize=(8, 6))
    plt.plot(xValues, initial_function, label='Початкова функція', color='blue')
    plt.plot(xValues, approx_function, label='Наближення рядом Фур\'є', color='red', linestyle='--')
    plt.title('Початкова функція та її наближення')
    plt.xlabel('x')
    plt.xlim(2,3)
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_error_and_save_to_file(xValues, N, aCoeffs, bCoeffs, filename="results.txt"):
    max_error = 0.0
    with open(filename, 'w') as f:
        f.write(f"N = {N}\n")
        f.write("Коефіцієнти a_k (всі = 0):\n")
        f.write(", ".join(map(str, aCoeffs)) + "\n")


        f.write("Коефіцієнти b_k:\n")
        f.write(", ".join(map(str, bCoeffs)) + "\n")

        for x in xValues:
            y = F(x)
            y_approx = FourierSeries(x, N)
            error = abs(y - y_approx)

            if error > max_error:
                max_error = error

        f.write(f"\nМаксимальна похибка: {max_error * 0.001:.9f}\n")

    print(f"Результати збережено у файл: {filename}")

def main():

    N = 30
    xValues = np.linspace(0, np.pi, 1000)  # Інтервал [0, π]

    # Обчислення коефіцієнтів
    aCoeffs = [A_k(k) for k in range(N + 1)]
    bCoeffs = [B_k(k) for k in range(1, N + 1)]

    # Побудова графіків гармонік
    plot_harmonics(aCoeffs, bCoeffs, N)

    # Графік початкової функції та її наближення
    plot_function_and_approximation(xValues, N)

    # Обчислення похибки та запис у файл
    calculate_error_and_save_to_file(xValues, N, aCoeffs, bCoeffs)

if __name__ == "__main__":
    main()
