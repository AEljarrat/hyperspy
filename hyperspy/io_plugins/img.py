# -*- coding: utf-8 -*-
# Copyright 2007-2016 The HyperSpy developers
#
# This file is part of  HyperSpy.
#
#  HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

import os

import numpy as np

# Plugin characteristics
# ----------------------
format_name = 'Signal2D'
description = 'Import/Export IMG format'
full_support = False
file_extensions = ['img', ]
default_extension = 0  # png


# Writing features
writes = [(2, 0), ]
# ----------------------


# TODO Extend it to support SI
def file_writer(filename, signal, file_format='png', **kwds):
    """Writes data to any format supported by PIL

        Parameters
        ----------
        filename: str
        signal: a Signal instance
        file_format : str
            The fileformat defined by its extension that is any one supported by
            PIL.
    """
    imsave(filename, signal.data)


def file_reader(filename, **kwds):
    """Read data from any format supported by PIL.

    Parameters
    ----------
    filename: str

    """
    dc, dx, dy = _read_data(filename, delay=False)
    lazy = kwds.pop('lazy', False)
    if lazy:
        # load the image fully to check the dtype and shape, should be cheap.
        # Then store this info for later re-loading when required
        from dask.array import from_delayed
        from dask import delayed
        val = delayed(_read_data, pure=True)(filename)
        dc = from_delayed(val, shape=dc.shape, dtype=dc.dtype)
    return [{'data': dc,
             'axes': [
                 {'name': 'y',
                  'size': dc.shape[0],
                  'offset': 0,
                  'scale': dy,
                  'units': 'nm'},
                 {'name': 'y',
                  'size': dc.shape[1],
                  'offset': 0,
                  'scale': dx,
                  'units': 'nm'} ],
             'metadata':
             {
                 'General': {'original_filename': os.path.split(filename)[1]},
                 "Signal": {'signal_type': "",
                            'record_by': 'image', },
             }
             }]

def _read_data(filename, delay=True):
    with open(filename, "rb") as f:
        header = np.fromfile(f,'int32',8)
        Nx = header[4]
        Ny = header[3]
        t = np.fromfile(f,'float64',1)
        dx = np.fromfile(f,'float64',1)[0]
        dy = np.fromfile(f,'float64',1)[0]
        # additional parameters
        paramSize = header[1]
        if (paramSize > 0):
            params = np.fromfile(f, 'float64', paramSize)
        # comments
        commtSize = header[2]
        if (commtSize > 0):
            commt = np.fromfile(f, 'char', paramSize)
        # dtype flag
        integerFlag = 0
        complexFlag = header[5]
        doubleFlag = (header[6] == 8*(complexFlag+1))
        flag = integerFlag+doubleFlag*4+complexFlag*2
        ##
        aa = np.bitwise_and(flag,7)
        if aa == 0:
            data = np.rollaxis(np.fromfile(f,'float32',Nx*Ny).reshape(Ny,Nx),1)
        elif aa == 1:
            data = np.rollaxis(np.fromfile(f,'float16',Nx*Ny).reshape(Ny,Nx),1)
        elif aa == 2:
            data = np.rollaxis(np.fromfile(f,'float32',2*Nx*Ny).reshape(Ny,2*Nx),1)
            data = data[0::2,:] +1j* data[1::2,:]
        elif aa == 4:
            data = np.rollaxis(np.fromfile(f,'float64',Nx*Ny).reshape(Ny,Nx),1)
        elif aa == 5:
            data = np.rollaxis(np.fromfile(f,'float32',Nx*Ny).reshape(Ny,Nx),1)
        elif aa == 6:
            data = np.rollaxis(np.fromfile(f,'float64',2*Nx*Ny).reshape(Ny,2*Nx),1)
            data = data[0::2,:] +1j* data[1::2,:]
    #return data and/or params depending on delay
    if delay:
        return data
    else:

        return data, dx, dy
