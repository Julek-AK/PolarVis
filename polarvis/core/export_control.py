
# Builtin
from dataclasses import dataclass
from typing import List
from pathlib import Path

# External

# Internal
from ..io.export import *
from ..io.formats import polar_to_stokes


@dataclass
class ExportConfig:
    results: List[str]
    save_directory: Path
    representation: str
    file_format: str


def convert_representation(array, representation):

    if representation == 'polar':
        return array

    if representation == 'stokes':
        return polar_to_stokes(array)

    raise ValueError(f'[Exporting] Unknown representation: {representation}')


def export_array(array, name, representation, directory, file_format):

    if file_format == 'CSV':
        export_csv(array, representation, directory / f'{name}_{representation}.csv')

    elif file_format == 'NPY':
        export_npy(array, representation, directory / f'{name}_{representation}.npy')

    elif file_format == 'MAT':
        export_mat(array, representation, directory / f'{name}_{representation}.mat')

    else:
        raise ValueError(f'[Exporting] Unknown export format: {file_format}')


def execute_export(config: ExportConfig, cache_manager) -> None:

    print(f"Exporting {len(config.results)} result(s)...")

    config.save_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for result_id in config.results:

            array = cache_manager.get_array(result_id)

            array = convert_representation(
                array,
                config.representation
            )
            
            print(f"  Exporting '{result_id}'")
            
            export_array(
                array,
                result_id,
                config.representation,
                config.save_directory,
                config.file_format
            )

            
    print(
        f"Successfully exported {len(config.results)} "
        f"result(s) to '{config.save_directory}'."
    )