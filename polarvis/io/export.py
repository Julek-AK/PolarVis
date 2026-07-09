
# Builtin
from pathlib import Path
import csv

# External
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat

# Internal



def export_npy(array: NDArray, representation: str, path: Path):
    np.save(path, array)


def export_csv(array: NDArray, representation: str, path: Path):
    h, w, _ = array.shape

    with open(path, 'w', newline='') as file:

        writer = csv.writer(file)

        if representation == 'polar':

            writer.writerow([
                'x',
                'y',
                'I',
                'DoLP',
                'AoP'
            ])

        elif representation == 'stokes':
            
            writer.writerow([
                'x',
                'y',
                'S0',
                'S1',
                'S2'
            ])

        else:
             raise ValueError(f'[Exporting] Unknown representation: {representation}')

        for y in range(h):
            for x in range(w):

                writer.writerow([
                    x,
                    y,
                    *array[y, x]
                ])

def export_mat(array: NDArray, representation: str, path: Path):

    if representation == 'polar':

        savemat(
            path,
            {
                'I': array[:, :, 0],
                'DoLP': array[:, :, 1],
                'AoP': array[:, :, 2]
            }
        )

    elif representation == 'stokes':

        savemat(
            path,
            {
                'S0': array[:, :, 0],
                'S1': array[:, :, 1],
                'S2': array[:, :, 2]
            }
        )

    else:
        raise ValueError(f'[Exporting] Unknown representation: {representation}')