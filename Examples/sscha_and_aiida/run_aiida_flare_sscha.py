"""Test for an actual AiiDA-FLARE-powered ensemble computation."""
import numpy as np

from ase.io import read
from ase.build import bulk, make_supercell
from ase.calculators.lj import LennardJones

from cellconstructor.Phonons import Phonons, compute_phonons_finite_displacements
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


def main():
    """Run with AiiDA-QuantumESPRESSO + FLARE + SSCHA @ NPT."""
    # =========== GENERAL INPUTS =============== #
    np.random.seed(0)
    number_of_configurations = 50
    batch_number = 1
    check_time = 3
    max_iterations = 10
    temperature = 0
    pressure = 0
    meaningful_factor = 0.5
    kong_liu_ratio = 0.5
    minimization_step = 0.1
    supercell = [2,2,2]

    atoms = read('./Si.pwi') # bulk('Si')
    structure = Structure()
    structure.generate_from_ase_atoms(atoms)

    # =========== FLARE MODEL =============== #
    flare_calc, _ = SGP_Calculator.from_file('./model.json')
    # flare_calc = SGP_Calculator(get_empty_sgp(n_types=1, the_map={14: 0}, the_atom_energies={0: -154.272015018195}))

    # =========== DYNAMICAL MATRIX =============== #
    dyn = compute_phonons_finite_displacements(structure, flare_calc, supercell=supercell)
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()
    
    # =========== AIIDA ENSEMBLE =============== #
    ensemble = AiiDAEnsemble(dyn, temperature)
    ensemble.set_otf(
        flare_calc, 
        std_tolerance_factor=-0.9, 
        max_atoms_added=-1, 
        update_threshold=0.5,
        update_style="threshold",
    )

    # =========== AiiDA INPUTS =============== #
    pw_code_label = 'pw@localhost'
    aiida_inputs = dict(
        pw_code=pw_code_label,
        protocol='fast',
        overrides={
            'clean_workdir': True,
            'meta_parameters':{'conv_thr_per_atom': 1e-8},
            'kpoints_distance': 0.4,
        },
        options={
            'resources':{'num_machines': 1, 'num_mpiprocs_per_machine': 4,},
            'prepend_text':'eval "$(conda shell.bash hook)"\nconda activate base\nexport OMP_NUM_THREADS=1',
        },
        electronic_type=ElectronicType.METAL,
        batch_number=batch_number,
        check_time=check_time,
    )

    # =========== SSCHA SETTINGS & COMPUTE =============== #
    minim = SSCHA_Minimizer(ensemble)
    minim.set_minimization_step(minimization_step)
    minim.kong_liu_ratio = kong_liu_ratio # default 0.5
    minim.meaningful_factor = meaningful_factor

    relax = SSCHA(
        minimizer=minim,
        aiida_inputs=aiida_inputs,
        N_configs=number_of_configurations,
        max_pop=max_iterations,
        save_ensemble=True,
    )

    ioinfo = IOInfo()
    ioinfo.SetupSaving(f'./minim_t{temperature}')
    relax.setup_custom_functions( custom_function_post = ioinfo.CFP_SaveAll )

    # Run the NPT simulation
    relax.vc_relax(
        target_press = pressure,
        restart_from_ens = False,
        ensemble_loc = f'./ensembles_P{pressure}_T{temperature}',
    )

    # Print in standard output
    relax.minim.finalize()


if __name__ == '__main__':
    main()

