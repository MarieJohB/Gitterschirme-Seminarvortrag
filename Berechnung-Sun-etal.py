
import numpy as np
import matplotlib.pyplot as plt

# Lichtgeschwindigkeit
c0 = 299792458  # m/s

# Berechnung der Wellenlänge in mm
# f_GHz kann ein einzelner Wert oder ein NumPy-Array sein
f_GHz = np.linspace(1.5, 4.5, 500)
lambda_mm = c0 / (f_GHz * 1e9) * 1e3


def einzelgitter(w, d, lambda_mm):
    """
    Berechnet die Schirmdaempfung (SE) und den Impedanzfaktor (gamma)
    eines einzelnen Gitters.

    Parameter:
        w        : Gitterweite
        d        : Material-/Drahtdurchmesser
        lambda_mm: Wellenlaenge in mm

    Rueckgabe:
        SE       : Schirmdaempfung in dB
        gamma    : Impedanzfaktor
    """

    wEff = w - 0.95 * d                    # Gl. (5)

    gitterfaktor = -np.log(-np.expm1(-np.pi * d / wEff))  # Gl. (4)

    z = 1j * (wEff / lambda_mm) * gitterfaktor

    T = 2 * z / (1 + 2 * z)               # Gl. (1)

    SE = -20 * np.log10(np.abs(T))         # Gl. (3)

    gamma = (1 - z) / (1 + z)             # Gl. (13)

    return SE, gamma


def doppelgitter(SE1, gamma1, SE2, gamma2, abstand, lambda_mm):
    """
    Berechnet die Schirmdaempfung eines Doppelgitters.

    Parameter:
        SE1, gamma1 : Schirmdaempfung und Impedanzfaktor Gitter 1
        SE2, gamma2 : Schirmdaempfung und Impedanzfaktor Gitter 2
        abstand     : Abstand zwischen den Gittern
        lambda_mm   : Wellenlaenge in mm

    Rueckgabe:
        SE          : Gesamte Schirmdaempfung in dB
    """

    phase = 4 * np.pi * abstand / lambda_mm

    M = 20 * np.log10(
        np.abs(1 - gamma1 * gamma2 * np.exp(-1j * phase))
    )                                           # Gl. (13)

    SE = SE1 + SE2 + M                         # Gl. (12)

    return SE


# ###################
#  Gitterparameter  #
# ###################

# Gitter fein
w_fein = 1.29       # Maschenweite in mm
d_fein = 0.3        # Drahtdurchmesser in mm

# Gitter grob (Problemsieb)
w_grob = 2.0        # Maschenweite in mm
d_grob = 0.5        # Drahtdurchmesser in mm

# Punktverschweißtes Gitter
w_punkt = 6.0       # Maschenweite in mm
d_punkt = 0.5       # Drahtdurchmesser in mm

# Abstand zwischen den Gittern
abstand = 5.0       # 0,5 cm = 5 mm

# Einzelgitter fein
SE_fein, gamma_fein = einzelgitter(
    w_fein, d_fein, lambda_mm
)

# Einzelgitter grob
SE_grob, gamma_grob = einzelgitter(
    w_grob, d_grob, lambda_mm
)

# Einzelgitter punktverschweißt
SE_punkt, gamma_punkt = einzelgitter(
    w_punkt, d_punkt, lambda_mm
)


# Doppelgitter fein (2 identische Gitter)
SE_doppel_fein = doppelgitter(
    SE_fein, gamma_fein,
    SE_fein, gamma_fein,
    abstand, lambda_mm
)

# Doppelgitter grob (2 identische Gitter)
SE_doppel_grob = doppelgitter(
    SE_grob, gamma_grob,
    SE_grob, gamma_grob,
    abstand, lambda_mm
)


###################
# Plots erstellen #
###################

plt.figure(figsize=(12, 7))

# Doppelgitter fein
plt.plot(
    f_GHz, SE_doppel_fein,
    color='blue',
    linestyle='-',
    linewidth=2,
    label='Doppelgitter fein'
)

# Einzelgitter fein
plt.plot(
    f_GHz, SE_fein,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Einzelgitter fein'
)

# Einzelgitter punktverschweißt
plt.plot(
    f_GHz, SE_punkt,
    color='gold',
    linestyle='--',
    linewidth=2,
    label='Einzelgitter punktverschweißt'
)

# Einzelgitter grob
plt.plot(
    f_GHz, SE_grob,
    color='purple',
    linestyle='--',
    linewidth=2,
    label='Einzelgitter grob'
)

# Doppelgitter grob
plt.plot(
    f_GHz, SE_doppel_grob,
    color='green',
    linestyle='-',
    linewidth=2,
    label='Doppelgitter grob'
)

plt.xlabel('Frequenz (GHz)', fontsize=12)
plt.ylabel('Schirmdämpfung SE (dB)', fontsize=12)

plt.title(
    'Vergleich der Schirmdämpfung verschiedener Gitterkonfigurationen',
    fontsize=14
)

plt.xlim(1.5, 4.5)
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()