"""Test for an actual AiiDA-FLARE-powered ensemble computation."""
import numpy as np

from ase.build import bulk, make_supercell
from ase.calculators.lj import LennardJones

from cellconstructor.Phonons import compute_phonons_finite_displacements
from cellconstructor.Structure import Structure
from sscha.aiida_ensemble import AiiDAEnsemble
from sscha.SchaMinimizer import SSCHA_Minimizer
from sscha.Relax import SSCHA
from sscha.Utilities import IOInfo

from aiida import load_profile
from aiida_quantumespresso.common.types import ElectronicType

from flare.bffs.sgp.calculator import SGP_Calculator
from get_sgp import get_empty_sgp

load_profile()

# PID: 1230420, 1292679

def main():
    """Run with AiiDA-QuantumESPRESSO + FLARE + SSCHA @ NVT."""
    # =========== GENERAL INPUTS =============== #
    np.random.seed(0)
    number_of_configurations = 8
    max_iterations = 20
    temperature = 0.0

    # =========== AiiDA ENSEMBLE =============== #
    atoms = bulk('Si')
    matrix = [[-1,1,1],[1,-1,1],[1,1,-1]] # ==> 8 atoms cell | i.e. conventional cell
    atoms = make_supercell(atoms, matrix)
    structure = Structure()
    structure.generate_from_ase_atoms(atoms)

    dyn = compute_phonons_finite_displacements(structure, LennardJones(), supercell=[1,1,1])
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    ensemble = AiiDAEnsemble(dyn, temperature)
    flare_calc = SGP_Calculator(get_empty_sgp(n_types=1, the_map={14: 0}, the_atom_energies={0: 0}))
    ensemble.set_otf(flare_calc, std_tolerance_factor=-0.001, max_atoms_added=-1)

    # =========== AiiDA INPUTS =============== #
    pw_code_label = 'pw@localhost'
    aiida_inputs = dict(
        pw_code=pw_code_label,
        protocol='fast',
        overrides={
            'meta_parameters':{'conv_thr_per_atom': 1e-8},
            'kpoints_distance': 0.8,
        },
        options={
            'resources':{'num_machines': 1, 'num_mpiprocs_per_machine': 1,},
            'prepend_text':'eval "$(conda shell.bash hook)"\nconda activate aiida-sscha\nexport OMP_NUM_THREADS=1',
        },
        electronic_type=ElectronicType.METAL,
    )

    # =========== SSCHA SETTINGS & COMPUTE =============== #
    minim = SSCHA_Minimizer(ensemble)
    minim.set_minimization_step(0.1)

    relax = SSCHA(
        minimizer=minim,
        aiida_inputs=aiida_inputs,
        N_configs=number_of_configurations,
        max_pop=max_iterations,
        save_ensemble=True,
    )

    ioinfo = IOInfo()
    ioinfo.SetupSaving('./minim_t0')
    relax.setup_custom_functions( custom_function_post = ioinfo.CFP_SaveAll)
    
    # Run the NVT simulation
    relax.vc_relax(
        target_press = 0.0,
        restart_from_ens = False,
        ensemble_loc = './ensembles_P0_T0',
    )

    # Print in standard output
    relax.minim.finalize()


if __name__ == '__main__':
    main()

