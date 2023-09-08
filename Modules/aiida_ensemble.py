# -*- coding: utf-8 -*-
"""Module for handling automated calculation via aiida-quantumespresso."""
from __future__ import annotations

from copy import copy, deepcopy
import time

from ase import units
from cellconstructor.Structure import Structure
import numpy as np
from numpy import ndarray

from .Ensemble import Ensemble

try:
    from aiida.orm import WorkChainNode
    from qe_tools import CONSTANTS

    gpa_to_rybohr3 = 1.0 / (CONSTANTS.ry_si / CONSTANTS.bohr_si**3 / 1.0e9)  # GPa -> Ry/Bohr^3
    ase_stress_units = -1.0 * gpa_to_rybohr3 * units.Ry / units.Bohr**3  # convention as in ASE (sign and eV/Ang^3)
except ImportError:
    import warnings
    warnings.warn('aiida or aiida-quantumespresso are not installed')

try:
    from flare.atoms import FLARE_Atoms
    from flare.learners.utils import get_env_indices, is_std_in_bound
except ImportError:
    pass


class AiiDAEnsemble(Ensemble):
    """Ensemble subclass to interface SSCHA with aiida-quantumespresso."""

    def compute_ensemble( # pylint: disable=arguments-renamed
        self,
        pw_code: str,
        protocol: str = 'moderate',
        options: dict = None,
        overrides: dict = None,
        group_label: str = None,
        **kwargs
    ) -> None:
        """Get ensemble properties.

        All the parameters refer to the
        :func:`aiida_quantumespresso.workflows.pw.base.PwBaseWorkChain.get_builder_from_protocol`
        method.

        Args:
        ----
            pw_code: The string associated with the AiiDA code for `pw.x`
            protocol: The protocol to be used; available protocols are 'fast', 'moderate' and 'precise'
            options: The options for the calculations, such as the resources, wall-time, etc.
            overrides: The overrides for the get_builder_from_protocol
            group_label: The group label where to add the submitted nodes for eventual future inspection
            kwargs: The kwargs for the get_builder_from_protocol

        """
        from aiida.orm import load_group

        group = None if group_label is None else load_group(group_label)

        # Check if not all the calculation needs to be done
        if self.force_computed is None:
            self.force_computed = np.array([False] * self.N, dtype=bool)

        n_calcs = np.sum(self.force_computed.astype(int))
        computing_ensemble = self

        self.has_stress = True  # by default we calculate stresses with the `get_builder_from_protocol`
        if overrides:
            try:
                tstress = overrides['pw']['parameters']['CONTROL']['tstress']
                self.has_stress = tstress
            except KeyError:
                pass

        # Check wheter compute the whole ensemble, or just a small part
        should_i_merge = False
        if n_calcs != self.N:
            should_i_merge = True
            computing_ensemble = self.get_noncomputed()
            self.remove_noncomputed()

        structures = copy(computing_ensemble.structures)
        dft_indices = np.arange(0, len(structures), 1).tolist()  # store here the indices to run with DFT/AiiDA

        # ============= FLARE SECTION ============= #
        # If a model is specified and it's not empty, try to predict.
        # Predict only the ones that are within uncertainty, the rest do via DFT/AiiDA.
        if self.gp_model is not None:
            if self.max_atoms_added < 0:
                self.max_atoms_added = structures[0].get_ase_atoms().get_global_number_of_atoms()
            if len(self.gp_model.training_data) > 0:
                self._predict_with_model(structures, computing_ensemble, dft_indices)

        # ============= AIIDA SECTION START ============= #
        workchains = submit_and_get_workchains(
            structures=[structures[i] for i in dft_indices],
            pw_code=pw_code,
            temperature=self.current_T,
            dft_indices=dft_indices,
            protocol=protocol,
            options=options,
            overrides=overrides,
            **kwargs
        )

        if group:
            group.add_nodes(workchains)

        workchains_copy = copy(workchains)
        while workchains_copy:
            workchains_copy = get_running_workchains(workchains_copy, computing_ensemble.force_computed)
            if workchains_copy:
                time.sleep(60)  # wait before checking again

        for i, is_computed in enumerate(computing_ensemble.force_computed):
            if is_computed:
                dft_stress = None
                wc = workchains[dft_indices.index(i)]
                dft_energy = wc.outputs.output_parameters.dict.energy
                dft_forces = wc.outputs.output_trajectory.get_array('forces')[-1]
                computing_ensemble.energies[i] = dft_energy / CONSTANTS.ry_to_ev
                computing_ensemble.forces[i] = dft_forces / CONSTANTS.ry_to_ev
                if self.has_stress:
                    stress = wc.outputs.output_trajectory.get_array('stress')[-1]
                    computing_ensemble.stresses[i] = stress * gpa_to_rybohr3
                    dft_stress = ase_stress_units * np.array([
                        stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]
                    ])

                if self.gp_model is not None:
                    self._update_gp(
                        FLARE_Atoms.from_ase_atoms(wc.inputs.pw.structure.get_ase()),
                        dft_frcs=dft_forces,
                        dft_energy=dft_energy,
                        dft_stress=dft_stress,
                    )
        # ============= AIIDA SECTION END ============= #

        if self.gp_model is not None:
            self._train_gp()
            self._write_model()

        # ============= FINALIZE ============= #
        if self.has_stress:
            computing_ensemble.stress_computed = copy(computing_ensemble.force_computed)

        print('CE BEFORE MERGE:', len(self.force_computed))

        if should_i_merge:
            self.merge(computing_ensemble)  # Remove the noncomputed ensemble from here, and merge
        print('CE AFTER MERGE:', len(self.force_computed))

    def _predict_with_model(
        self,
        structures: list[Structure],
        computing_ensemble: Ensemble,
        dft_indices: list[int],
    ) -> None:
        """Predict on all the structures and estimate errors.

        This is used to remove the structures indecis to not compute via AiiDA/DFT.

        Args:
        ----
            structures: list of :class:`~cellconstructor.Structure.Structure` to simulate
            computing_ensemble: the :class:`~sscha.Ensemble` with the forces to compute
            dft_indices: list of integers related to the structures

        """
        for index, structure in enumerate(structures):
            atoms = FLARE_Atoms.from_ase_atoms(structure.get_ase_atoms())
            self._compute_properties(atoms)

            # get max uncertainty atoms
            if self.build_mode == 'bayesian':
                env_selection = is_std_in_bound
            elif self.build_mode == 'direct':
                env_selection = get_env_indices

            tic = time.time()

            std_in_bound, _ = env_selection(
                self.std_tolerance,
                self.gp_model.force_noise,
                atoms,
                max_atoms_added=self.max_atoms_added,
                update_style=self.update_style,
                update_threshold=self.update_threshold,
            )

            self.output.write_wall_time(tic, task='Env Selection')

            if std_in_bound:
                dft_indices.remove(index)  # remove index computed via ML-FF

                computing_ensemble.energies[index] = atoms.potential_energy / units.Ry
                computing_ensemble.forces[index] = deepcopy(atoms.forces) / units.Ry
                if self.has_stress:
                    computing_ensemble.stresses[index] = -1 * deepcopy(
                        atoms.get_stress(voigt=False)
                    ) * units.Bohr**3 / units.Ry

                computing_ensemble.force_computed[index] = True

    def _compute_properties(self, atoms: FLARE_Atoms) -> None:
        """Compute energies, forces, stresses, and their uncertainties.

        The FLARE ASE calculator is used, and write the results.

        Args:
        ----
            atoms: a :class:`flare.atoms.FLARE_Atoms` instance for which to compute properties

        """
        tic = time.time()

        atoms.calc = self.flare_calc
        atoms.calc.calculate(atoms)

        self.output.write_wall_time(tic, task='Compute Properties')

    def _write_model(self) -> None:
        """Write the current model in a JSON file."""
        self.flare_calc.write_model(self.flare_name)

    def _update_gp(
        self,
        atoms: FLARE_Atoms,
        dft_frcs: ndarray,
        dft_energy: float | None = None,
        dft_stress: ndarray | None = None,
    ) -> None:
        """Update the current GP model.

        Args:
        ----
            atoms (FLARE_Atoms): :class:`flare.atoms.FLARE_Atoms`` instance whose
                local environments will be added to the training set.
            dft_frcs (np.ndarray): DFT forces on all atoms in the structure, in eV/Angstrom.
            dft_energy (float): total energy of the entire structure, in eV.
            dft_stress (np.ndarray): DFT forces on all atoms in the structure.
                Sign as in ASE (-1 in respect with QE), units in eV/Angstrom^3,
                and in Voigt notation, i.e. (xx, yy, zz, yz, xz, xy).

        """
        from ase.calculators.singlepoint import SinglePointCalculator

        tic = time.time()

        self._compute_properties(atoms)

        # get max uncertainty atoms
        if self.build_mode == 'bayesian':
            env_selection = is_std_in_bound
        elif self.build_mode == 'direct':
            env_selection = get_env_indices

        tic = time.time()

        std_in_bound, train_atoms = env_selection(
            self.std_tolerance,
            self.gp_model.force_noise,
            atoms,
            max_atoms_added=self.max_atoms_added,
            update_style=self.update_style,
            update_threshold=self.update_threshold,
        )

        self.output.write_wall_time(tic, task='Env Selection')

        # Here we make the decision to skip adding environments even if the
        # DFT calculation was performed. This avoids slowing down the model,
        # while the SSCHA is feeded with the DFT results.
        if not std_in_bound:
            stds = self.flare_calc.results.get('stds', np.zeros_like(dft_frcs))
            self.output.add_atom_info(train_atoms, stds)

            # Convert ASE stress (xx, yy, zz, yz, xz, xy) to FLARE stress
            # (xx, xy, xz, yy, yz, zz).
            flare_stress = None
            if dft_stress is not None:
                flare_stress = -np.array([
                    dft_stress[0],
                    dft_stress[5],
                    dft_stress[4],
                    dft_stress[1],
                    dft_stress[3],
                    dft_stress[2],
                ])

            results = {
                'forces': atoms.forces,
                'energy': atoms.potential_energy,
                'free_energy': atoms.potential_energy,
                'stress': atoms.stress,
            }

            atoms.calc = SinglePointCalculator(atoms, **results)

            # update gp model
            self.gp_model.update_db(
                atoms,
                dft_frcs,
                custom_range=train_atoms,
                energy=dft_energy,
                stress=flare_stress,
            )

            self.gp_model.set_L_alpha()
            self.output.write_wall_time(tic, task='Update GP')

            # write model
            self._write_model()

    def _train_gp(self) -> None:
        """Optimize the hyperparameters of the current GP model."""
        tic = time.time()

        self.gp_model.train(logger_name=self.output.basename + 'hyps')

        self.output.write_wall_time(tic, task='Train Hyps')

        hyps, labels = self.gp_model.hyps_and_labels
        if labels is None:
            labels = self.gp_model.hyp_labels

        self.output.write_hyps(
            labels,
            hyps,
            tic, # actually here there should be the actual start time of the entire simulation
            self.gp_model.likelihood,
            self.gp_model.likelihood_gradient,
            hyps_mask=self.gp_model.hyps_mask,
        )


def get_running_workchains(workchains: list[WorkChainNode], success: list[bool]) -> list:
    """Get the running workchains popping the finished ones.

    Two extra array should be given to populate the successfully finished runs.

    Args:
    ----
        workchains: list of :class:`~aiida.orm.WorkChainNode`
        success: list where to store whether the workchains finished successfully or not.

    """
    wcs_left = copy(workchains)

    for workchain in workchains:
        if workchain.is_finished:
            if workchain.is_failed:
                print(f'[FAILURE] for <PwBaseWorkChain> with PK={workchain.pk}')
            else:
                index = int(workchain.label.split('_')[-1])
                success[index] = True
                print(f'[SUCCESS] for <PwBaseWorkChain> with PK={workchain.pk}')

            wcs_left.remove(workchain)  # here it may be critical

    return wcs_left


def submit_and_get_workchains(
    structures: list[Structure],
    pw_code: str,
    temperature: float | int,
    dft_indices: list[int],
    protocol: str = 'moderate',
    options: dict = None,
    overrides: dict = None,
    **kwargs
) -> list[WorkChainNode]:
    """Submit and return the workchains for a list of :class:`~cellconstructor.Structure.Structure`.

    Args:
    ----
        structures: a list of :class:`~cellconstructor.Structure.Structure` to run via PwBaseWorkChain.
        pw_code: The string associated with the AiiDA code for `pw.x`
        temperature: The temperature corresponding to the structures ensemble
        dft_indices: The indices of the compute ensemble related to the structures.
        protocol: The protocol to be used; available protocols are 'fast', 'moderate' and 'precise'
        options: The options for the calculations, such as the resources, wall-time, etc.
        overrides: The overrides for the get_builder_from_protocol
        kwargs: The kwargs for the get_builder_from_protocol

    """
    from aiida.engine import submit
    from aiida.orm import StructureData
    from aiida.plugins import WorkflowFactory

    PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')

    structures_data = [StructureData(ase=cc.get_ase_atoms()) for cc in structures]
    workchains = []

    for i, structure in zip(dft_indices, structures_data):
        builder = PwBaseWorkChain.get_builder_from_protocol(
            code=pw_code, structure=structure, protocol=protocol, options=options, overrides=overrides, **kwargs
        )
        builder.metadata.label = f'T_{temperature}_id_{i}'
        workchains.append(submit(builder))
        print(f'Launched <PwBaseWorkChain> with PK={workchains[-1].pk}')

    return workchains
