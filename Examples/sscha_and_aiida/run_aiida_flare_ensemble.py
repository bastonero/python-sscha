"""Test for an actual AiiDA-FLARE-powered ensemble computation."""
import numpy as np

from ase import Atoms
from ase.build import make_supercell
from ase.calculators.lj import LennardJones

from cellconstructor.Phonons import compute_phonons_finite_displacements
from cellconstructor.Structure import Structure
from sscha.aiida_ensemble import AiiDAEnsemble

from aiida import load_profile
from aiida_quantumespresso.common.types import ElectronicType

from flare.bffs.sgp.calculator import SGP_Calculator
from get_sgp import get_empty_sgp

load_profile()


def main():
    """Run with AiiDA-QuantumESPRESSO + FLARE some ensemble configuration for testing."""
    # =========== GENERAL INPUTS =============== #
    np.random.seed(0)
    number_of_configurations = 10
    batch_number = 3
    check_time = 3
    temperature = 0.0

    # =========== AiiDA ENSEMBLE =============== #
    a, sc_size, numbers = 2.0, 1, [6, 8]
    cell = np.eye(3) * a
    positions = np.array([[0, 0, 0], [0.5 , 0.5, 0.5]])
    unit_cell = Atoms(cell=cell, scaled_positions=positions, numbers=numbers, pbc=True)
    multiplier = np.identity(3) * sc_size
    atoms = make_supercell(unit_cell, multiplier)

    structure = Structure()
    structure.generate_from_ase_atoms(atoms)

    dyn = compute_phonons_finite_displacements(structure, LennardJones(), supercell=[1,1,1])
    dyn.Symmetrize()
    dyn.ForcePositiveDefinite()

    ensemble = AiiDAEnsemble(dyn, temperature)
    flare_calc = SGP_Calculator(get_empty_sgp())
    ensemble.set_otf(flare_calc)

    # =========== AiiDA INPUTS =============== #
    pw_code_label = 'pw@localhost'
    aiida_inputs = dict(
        pw_code=pw_code_label,
        protocol='fast',
        overrides={
            'meta_parameters':{'conv_thr_per_atom': 1e-6},
            'kpoints_distance': 1000
        },
        options={
            'resources':{'num_machines': 1, 'num_mpiprocs_per_machine': 2,},
            'prepend_text':'eval "$(conda shell.bash hook)"\nconda activate aiida-sscha\nexport OMP_NUM_THREADS=1',
        },
        electronic_type=ElectronicType.INSULATOR,
        batch_number=batch_number,
        check_time=check_time,
    )

    # =========== GENERATE & COMPUTE =============== #
    ensemble.generate(number_of_configurations)
    ensemble.compute_ensemble(**aiida_inputs) # this should include the training too

    print()
    print()
    print("=============================================")
    print("First population has run.")
    print("=============================================")
    print()
    print()

    ensemble.generate(number_of_configurations)  # here hopefully the model is called
    ensemble.compute_ensemble(**aiida_inputs)


if __name__ == '__main__':
    main()

