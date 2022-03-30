import numpy as np
from scipy.special import jv
from scipy.integrate import quad

mrad2Tomm2At100m = (1e-3*100)**2


def undulator_K (bfield, period):
    """
    Get the K parameter for undulator

    bfield : float - field in (T)
    period : fload - period in (m)
    """

    return 93.36 * bfield * period

def undulator_bfield (K, period):
    """
    Get the bfield for undulator


    K : float - undulator parameter
    period : fload - period in (m)
    """

    return K / (93.36 * period)

def undulator_period (K, bfield):
    """
    Get the period for undulator


    K : float - undulator parameter
    bfield : float - bfield in (T)
    """

    return K / (93.36 * bfield)



def undulator_energy (K, period, energy_GeV, harmonic=1):
    """
    Get the energy for a harmoic

    K : float - undulator parameter
    period : fload - period in (m)
    energy_GeV : float - electron beam energy
    harmonic : int - undulator harmonic
    """

    return 9.50 * energy_GeV**2 / ((1 + K**2/2)*period) * harmonic


def undulator_K_fromphoton (energy_eV, period, energy_GeV, harmonic=1):
    """
    Get the undulator K needed for specific photon energy

    energy_eV : float - photon energy (eV)
    period : fload - period in (m)
    energy_GeV : float - electron beam energy
    harmonic : int - undulator harmonic
    """

    return (2 * (9.50*energy_GeV**2*harmonic/(energy_eV*period) - 1))**0.5


def undulator_power (bfield, length, energy_GeV, current):
    """
    Get the power from undulator in Watts

    bfield : float - peak b bfield in T assuming sin
    length : float - length of undulator in m
    energy_GeV : float - electon beam energy in GeV
    current : float - electron beam current in Amps
    """

    return 633.0 * energy_GeV**2 * bfield**2 * length * current




def BesselK_IntegralToInfty (nu, x):
    # Compute the modified bessel function according to eqn 15 of:
    #   VACLAV O. KOSTROUN
    #   SIMPLE NUMERICAL EVALUATION OF MODIFIED BESSEL FUNCTIONS Kv(x) OF FRACTIONAL ORDER AND THE INTEGRAL fxKv(rT)dr7
    #   Nuclear Instruments and Methods 172 (1980) 371-374

    # The interval (0.5 by authros suitable)
    h = 0.5

    # Required precision of the r-th term.  Authors use 1e-5, we'll use higher
    epsilon = 1e-15

    # This is the 0-th term
    RthTerm = np.exp(-x) / 2. * h

    # This is the return value, at the moment only containing the 0-th term
    K = RthTerm

    # r is the summation index (ie the r-th term)
    r = 0

    # Continue until the r-th term satisfies the precision requirement
    # (Prefer to sum from small to large, but ok)
    while np.any(RthTerm > epsilon):

         # Increment r
        r += 1

        # Calculate the r-th term
        RthTerm = np.exp(-x * np.cosh(r * h)) * np.cosh(nu * r * h) / np.cosh(r * h)

        # Add r-th term to return vvalue
        K += RthTerm * h

    # Return the value of the modified bessel function
    return K



def Fn (n, K):
    """
    n : int (odd) - harmonic number
    K : float - undulator deflection parameter
    """

    return (K**2 * n**2) / (1 + K**2/2)**2 * (jv((n-1)/2, n * K**2 / (4*(1+K**2/2))) - jv((n+1)/2, n * K**2 / ((4*(1+K**2/2)))))**2



def G (K):
    return K/(1+K**2)**(7/2) * (K**6 + (24/7)*K**4 + 4*K**2 +(16/7))


def Fk (K, gtheta, gpsi):

    def integrand (alpha):
        D = 1 + gpsi**2 + (gtheta - K * np.cos(alpha))**2
        return (1/D**3 - 4*(gtheta - K * np.cos(alpha))**2 / D**5) * np.sin(alpha)**2

    return 16*K/(7*np.pi*G(K)) * quad(integrand, -np.pi, np.pi)[0]

def FkEPU (K, gtheta):

    def integrand (alpha):
        D = 1 + K**2 + gtheta**2 - 2*gtheta*K * np.cos(alpha)
        return (1/D**3 - 4 * (gtheta * np.sin(alpha))**2 / D**5)

    return 16*K/(7*np.pi*G(K)) * quad(integrand, -np.pi, np.pi)[0]


def undulator_power_density (bfield, period, length, energy_GeV, current, theta=0, psi=0):
    """
    get the undulator power density at any angle in W/rad^2

    bfield : float - effective magnetic field (T)
    period : float - undulator period (m)
    length : float - undulator length (m)
    energy_GeV : float - electron beam energy (GeV)
    current : float - electron beam current (A)
    theta : float - horizontal angle (rad)
    psi : float - vertical angle (rad)
    """

    K = undulator_K(bfield, period)
    gamma = energy_GeV / 0.511e-3
    N = int(length*2/period)/2.0

    return 10.84 * bfield * energy_GeV**4 * current * N * G(K) * Fk(K, gamma*theta, gamma*psi)


def epu_power_density (bfield, period, length, energy_GeV, current, theta=0):
    """
    get the undulator power density at any angle in W/rad^2

    bfield : float - effective magnetic field (T)
    period : float - undulator period (m)
    length : float - undulator length (m)
    energy_GeV : float - electron beam energy (GeV)
    current : float - electron beam current (A)
    theta : float - horizontal angle (rad)
    """

    K = undulator_K(bfield, period)
    gamma = energy_GeV / 0.511e-3
    N = int(length*2/period)/2.0

    return 10.84 * bfield * energy_GeV**4 * current * N * G(K) * FkEPU(K, gamma*theta)

