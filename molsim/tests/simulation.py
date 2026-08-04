import numpy as np

from molsim.file_handling import load_mol
from molsim.classes import Source

import sys

def get_test_mol():
    return load_mol('H2CN.cat', type='SPCAT')

def get_test_source():
    return Source(dV=0.3,velocity=5.8,Tex=8.,column=10**12)

def get_test_limits():
    ll = np.array([  3944.6407,  4347.00123,  8277.59218,  8462.60987,  8647.62613,  8832.64239,
                    9017.65865,  9202.70065,  9395.00133, 11052.51719, 11237.55061, 11422.55543,
                   12701.42731, 14352.50102, 18001.27472, 29272.13289, 29457.15773, 29642.17399,
                   29827.12018, 30012.13358, 30197.14841, 30382.15895, 30567.08369, 30752.10567,
                   30937.12908, 31122.15106, 31307.08581, 31492.10207, 31677.11833, 31862.13316,
                   32047.06219, 32232.08274, 32417.10329, 32602.12241, 32787.04286, 32972.06627,
                   33157.08825, 33342.12739, 35686.00893,])
    ul = np.array([ 4345.9988 ,  8277.17462,  8462.19946,  8647.2243 ,  8832.24914,  9017.27398,
                    9202.29882,  9324.99997, 11052.20688, 11237.1402 , 11422.14645, 11607.15413,
                   14348.99895, 15586.03332, 29271.70103, 29456.73302, 29641.75071, 29826.77412,
                   30011.71316, 30196.72084, 30381.73138, 30566.74621, 30751.67524, 30936.69722,
                   31121.72063, 31306.74404, 31491.67736, 31676.68933, 31861.70416, 32046.72042,
                   32231.65088, 32416.67143, 32601.69198, 32786.71253, 32971.64156, 33156.65925,
                   33341.65978, 34250.67646, 36410.59279,])

    return ll, ul

def test_simulation_py():
    from molsim.classes import Simulation as Sim

    mol = get_test_mol()
    src = get_test_source()
    ll, ul = get_test_limits()

    sim = Sim(ll=ll, ul=ul, mol=mol, source=src)

    # Used to save reference data
    # np.savez_compressed('reference.npz', freq0=sim.spectrum.freq0, frequency=sim.spectrum.frequency,  tau=sim.spectrum.tau,
    #                     Ibg=sim.spectrum.Ibg, Tbg=sim.spectrum.Tbg, Tb=sim.spectrum.Tb, Iv=sim.spectrum.Iv,
    #                     freq_profile=sim.spectrum.freq_profile, Tbg_profile=sim.spectrum.Tbg_profile, 
    #                     tau_profile=sim.spectrum.tau_profile, int_profile=sim.spectrum.int_profile)

    refdict = np.load('reference.npz', allow_pickle=False)

    atol = 1e-20
    rtol = 1e-14

    for key, refarray in refdict.items():
        calcarray = eval("sim.spectrum."+key)
        if not np.allclose(refarray, calcarray, rtol=rtol, atol=atol, equal_nan=True):
            np.set_printoptions(threshold=sys.maxsize)
            print("Reference:")
            print(refarray)
            print("Absolute Value of Difference:")
            adiff = np.abs(refarray-calcarray)
            print(adiff)
            m = np.argmax(adiff)
            print(f"Max Absolute Value of Difference at {m}:")
            print(f"Ref:  {refarray[m]:0.8e}")
            print(f"Calc: {calcarray[m]:0.8e}")
            print(f"Diff: {calcarray[m]-refarray[m]}")
            check = np.abs(refarray)*rtol+atol
            for i in range(check.size):
                if adiff[i] > check[i]:
                    print(f"Relative Difference problem at :", i)
                    print(f"Ref:      {refarray[i]:0.8e}")
                    print(f"Calc:     {calcarray[i]:0.8e}")
                    print(f"Diff:     {calcarray[i]-refarray[i]}")
                    print(f"Rel diff: {adiff[i]/refarray[i]}")
            raise AssertionError(f"Failed to match reference data for {key}!")

def test_simulation_cpp():
    from molsim.molsim_cpp import Simulation as Sim

    mol = get_test_mol()
    src = get_test_source()
    ll, ul = get_test_limits()

    sim = Sim(ll=ll, ul=ul, mol=mol, source=src, line_profile="gaussian")

    refdict = np.load('reference.npz', allow_pickle=False)

    atol = 1e-20
    rtol = 1e-14

    for key, refarray in refdict.items():
        if (key == "Iv"): # not yet implemented
            continue
        calcarray = eval("sim.spectrum."+key)
        if not np.allclose(refarray, calcarray, rtol=rtol, atol=atol, equal_nan=True):
            np.set_printoptions(threshold=sys.maxsize)
            print("Reference:")
            print(refarray)
            print("Computed values:")
            print(calcarray)
            print("Absolute Value of Difference:")
            adiff = np.abs(refarray-calcarray)
            print(adiff)
            m = np.argmax(adiff)
            print(f"Max Absolute Value of Difference at {m}:")
            print(f"Ref:  {refarray[m]:0.8e}")
            print(f"Calc: {calcarray[m]:0.8e}")
            print(f"Diff: {calcarray[m]-refarray[m]}")
            check = np.abs(refarray)*rtol+atol
            for i in range(check.size):
                if adiff[i] > check[i]:
                    print(f"Relative Difference problem at :", i)
                    print(f"Ref:      {refarray[i]:0.8e}")
                    print(f"Calc:     {calcarray[i]:0.8e}")
                    print(f"Diff:     {calcarray[i]-refarray[i]}")
                    print(f"Rel diff: {adiff[i]/refarray[i]}")

            raise AssertionError(f"Failed to match reference data for {key}!")



if __name__ == '__main__':
    test_simulation_py()
    test_simulation_cpp()
