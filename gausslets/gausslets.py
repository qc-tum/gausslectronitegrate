"""
Evaluate Gausslet coefficients.
"""

import numpy as np


def elementary_gaussian(x: float, j: int) -> float:
    """
    Evaluate an elementary Gaussian.
    """
    return np.exp(-0.5 * (3*x - j)**2)


def evaluate_gausslet_1d(b_list, j_list, x):
    """
    Evaluate a Gausslet function defined via the
    provided coefficients and indices in one dimension.
    """
    return sum(b * elementary_gaussian(x, j) for b, j in zip(b_list, j_list))


def evaluate_gausslet_3d(b_list, j_list, r):
    """
    Evaluate a Gausslet function defined via the
    provided coefficients and indices in three dimensions.
    """
    x, y, z = r
    return evaluate_gausslet_1d(b_list, j_list, x) \
         * evaluate_gausslet_1d(b_list, j_list, y) \
         * evaluate_gausslet_1d(b_list, j_list, z)


def get_gausslet_g10_coeffs():
    """
    Coefficients `b_j` of the Gaussians defining the Gausslet G10
    (Table V in "Hybrid grid/basis set discretizations of the Schrödinger equation").
    """
    b_half = [
         0.6006282292783031,
         0.3870904132059249,
         0.1167436095101837,
        -0.1401141978512072,
        -0.1178552983794614,
        -0.0112632618094700,
         0.0450560144981757,
         0.0502131666992306,
        -0.0207372799495982,
        -0.0031814624464224,
        -0.0214900136942583,
         0.0139369308627208,
        -0.0029594340072233,
         0.0057046712233152,
        -0.0026819334185882,
        -0.0004611902203357,
         0.0003205662299202,
        -0.0009695161114260,
         0.0012381620748654,
        -0.0008657512270795,
         0.0007050590750442,
        -0.0005322979066705,
         0.0003332874495659,
        -0.0002178032104139,
         0.0001389608411184,
        -0.0000849543923289,
         0.0000533515010750,
        -0.0000327971054166,
         0.0000193278214075,
        -0.0000108674604171,
         0.0000060213353043,
        -0.0000033140396282,
         0.0000016801358258,
        -0.0000008242534887,
         0.0000004139991910,
        -0.0000001869527576,
         0.0000000787310449,
        -0.0000000360812189,
         0.0000000168525628,
        -0.0000000087040883,
         0.0000000042255242,
        -0.0000000014639246,
         0.0000000006473451,
        -0.0000000003739856,
         0.0000000001419863,
        -0.0000000000705564,
         0.0000000000418054,
        -0.0000000000097639,
        -0.0000000000008755,
         0.0000000000004776,
        -0.0000000000000061,
        -0.0000000000000030,
        -0.0000000000000088,
         0.0000000000000063,
        -0.0000000000000019,
         0.0000000000000030,
        -0.0000000000000018,
         0.0000000000000002,
        -0.0000000000000001,
        -0.0000000000000004,
         0.0000000000000005,
        -0.0000000000000004,
         0.0000000000000003,
        -0.0000000000000003,
         0.0000000000000001,
        -0.0000000000000001,
         0.0000000000000000,
        -0.0000000000000001,
    ]
    return np.array(list(reversed(b_half[1:])) + b_half), np.arange(-len(b_half) + 1, len(b_half))
