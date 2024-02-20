# importation des modules utilisés
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Liste des fonctions du module


def List_Functions():
    print("GenerateSinImage(shape, u, v, plot=False)", GenerateSinImage.__doc__)
    print("------------ \n")
    print("Magnitude_Spectrum(image, plot=False)", Magnitude_Spectrum.__doc__)
    print("------------ \n")
    print("Phase_Spectrum(image, plot=False)", Phase_Spectrum.__doc__)
    print("------------ \n")
    print("IDFT(magnitude, phase, plot=False)", IDFT.__doc__)
    print("------------ \n")


# Fonctions locales
def GenerateSinImage(shape, u, v, plot=False):
    """
    Renvoie la matrice de pixels d'une image sinusoïdale
    en niveaux de gris de fréquence (u;v)
    """

    image = np.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            image[i, j] = (
                127 * np.cos(2 * np.pi * (u * i / shape[0] + v * j / shape[1])) + 128
            )

    if plot:
        plt.figure()
        plt.title(f"Image sinusoïdale en niveaux de gris de fréquence ({u};{v})")
        plt.imshow(image, cmap="gray")
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return image


def Get_DFT_shift(image):
    """
    Renvoie le tableau complexe contenant la transformée de Fourier à 2D de l'image
    """

    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift


def Magnitude_Spectrum(image, plot=False):
    """
    Renvoie la matrice de pixels contenant le spectre en amplitude de l'image
    """

    dft_shift = Get_DFT_shift(image)
    magnitude_spectrum = cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    if plot:
        magnitude_spectrum_dB = 20 * np.log(magnitude_spectrum)
        plt.figure()
        plt.subplot(121), plt.imshow(image, cmap="gray")
        plt.title("Input Image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(magnitude_spectrum_dB, cmap="gray")
        plt.title(r"Magnitude Spectrum $(dB)$")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return magnitude_spectrum


def Phase_Spectrum(image, plot):
    """
    Renvoie la matrice de pixels contenant le spectre en phase de l'image
    """

    dft_shift = Get_DFT_shift(image)
    phase_spectrum = cv.phase(dft_shift[:, :, 0], dft_shift[:, :, 1])

    if plot:
        plt.figure()
        plt.subplot(121), plt.imshow(image, cmap="gray")
        plt.title("Input Image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(phase_spectrum, cmap="gray")
        plt.title(r"Phase Spectrum $(dB)$")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return phase_spectrum


def IDFT(magnitude, phase, plot=False):
    """
    Reconstitue une image en niveaux de gris à partir de son spectre
    """

    real, imag = cv.polarToCart(magnitude, phase)
    output = cv.merge([real, imag])
    output_shift = np.fft.ifftshift(output)

    image_back = cv.idft(output_shift)
    image_back = cv.magnitude(image_back[:, :, 0], image_back[:, :, 1])
    image_back = cv.normalize(
        image_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )

    if plot:
        plt.figure()
        plt.imshow(image_back, cmap="gray")
        plt.title("Reconstructed Image")
        plt.colorbar()
        plt.show()

    return image_back
