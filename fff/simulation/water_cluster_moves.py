"""
This file contains a collection of functions for manipulating water cluster geometries. There are, of course,
more things you could do to a water cluster geometry. I have only implemented manipulations of the cartesian
coordinates. Things like permuting hydrogen atoms based on graph connectivities could also be implemented.

All of the functions operate on a single numpy array which is given as the first argument to the function.
Most of the functions take triplets of indices and operate on those indices in place rather than returning
a new numpy array.

Authored by Kristina Herman: https://github.com/HenrySprueill/fast-finedtuned-forcefields/blob/main/utils/water_cluster_moves.py
"""

import numpy as np
import math


def rotation_matrix(axis: np.ndarray, theta: float):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Taken from: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate_around_axis(geom: np.ndarray, axis: np.ndarray, theta: float):
    """
    Rotates all atoms in geom around an arbitrary axis by theta radians.
    NOTE: this does a rotation in so-called world space. This means, if you pass an axis
    such as a bond vector, it will not do what you want. You have to translate the geometry to the origin
    and the translate it back in order for that to work.

    geom: Numpy array representing the cartesian coordinates a group of atoms (or single atom)
    axis: 3 element List or Numpy array which specifies an axis around which to rotate. Does not have to be normalized.
    theta: The angle by which we should rotate in radians
    """
    R = rotation_matrix(axis, theta)

    geom_out = np.zeros_like(geom)
    # probably possible to vectorize this with an axis argument to numpy.
    for i in range(np.shape(geom)[0]):
        geom_out[i, :] = np.dot(R, geom[i, :])
    return geom_out


def rotate_around_HOH_bisector_axis(geom: np.ndarray, theta: float):
    '''
    Rotates around the bisector axis of a water molecule as specified by geom.
    I don't check that a valid water molecule is passed in, but this expects an OHH ordered water molecule.
    If you want to rotate multiple molecules around their local bisectors, you must call this multiple times.

    geom: Numpy array representing the cartesian coordinates of a water cluster
    theta: The angle by which we should rotate in radians
    '''
    original_O_position = geom[0, :].copy()
    # print(original_O_position)
    geom -= original_O_position  # translate to origin
    # print(geom)
    HOH_bisector = 0.5 * (geom[1, :] + geom[2, :])
    geom = rotate_around_axis(geom, HOH_bisector, theta)
    # print(geom)
    geom += original_O_position  # translate back to orignal position
    return geom


def rotate_around_local_axis(geom: np.ndarray, atom_id_1: int, atom_id_2: int, theta: float):
    '''
    Rotates around the axis going from atom_id_1 atom_id_2.
    Notice that these indices correspond to indices into geom, which will likely not be the same as indices
    into the entire water cluster.

    For instance, if you want to rotate around the OH bond axis, you would pass 0,1.
    If you want to rotate around the O-O axis of a dimer you pass in, you would give 0,4.

    I don't check that a valid water molecule is passed in, but this expects an OHH ordered water molecule.

    geom: Numpy array representing the cartesian coordinates of a water molecule or collection of water molecules.
    theta: The angle by which we should rotate in radians
    '''
    original_origin_position = np.copy(geom[atom_id_1, :])
    geom -= original_origin_position  # translate to origin
    rotation_axis = (geom[atom_id_2, :] - geom[atom_id_1, :]) / np.linalg.norm(geom[atom_id_2, :] - geom[atom_id_1, :])
    geom = rotate_around_axis(geom, rotation_axis, theta)
    geom += original_origin_position  # translate back to orignal position
    return geom
