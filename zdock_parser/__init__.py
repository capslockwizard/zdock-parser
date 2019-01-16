import numpy as np
import MDAnalysis
from numba import jit, float32
import cStringIO
import drsip_common as common
import types
import errno

@jit(float32[:,::1](float32), nopython=True)
def z_rot_func(rad):
    trans_rot = np.zeros((3,3), dtype=np.float32)
    trans_rot[0,:] = [np.cos(rad), -np.sin(rad), 0.0]
    trans_rot[1,:] = [np.sin(rad), np.cos(rad), 0.0]
    trans_rot[2,:] = [0.0, 0.0, 1.0]

    return trans_rot

@jit(float32[:,::1](float32), nopython=True)
def x_rot_func(rad):
    trans_rot = np.zeros((3,3), dtype=np.float32)
    trans_rot[0,:] = [1.0, 0.0, 0.0]
    trans_rot[1,:] = [0.0, np.cos(rad), -np.sin(rad)]
    trans_rot[2,:] = [0.0, np.sin(rad), np.cos(rad)]

    return trans_rot

@jit(float32[:,::1](float32[::1]), nopython=True)
def euler_to_rot_mat(rot_angles):
    return np.dot(np.dot(z_rot_func(rot_angles[2]), x_rot_func(rot_angles[1])), z_rot_func(rot_angles[0]))

@jit(float32[:,::1](float32[::1],float32[:,::1]), nopython=True)
def rot_with_euler_rot(rot_angles, coord):
    return np.dot(coord, euler_to_rot_mat(rot_angles).T)


def _read_next_timestep(self, ts=None):
    """copy next frame into timestep"""

    if self.ts.frame >= self.n_frames-1:
        raise IOError(errno.EIO, 'trying to go over trajectory limit')
    if ts is None:
        ts = self.ts
    ts.frame += 1

    self.zdock_inst._set_pose_num(ts.frame+1)
    ts._pos = self.zdock_inst.static_mobile_copy_uni.trajectory.ts._pos

    return ts


class ZDOCK(object):
    """Parse ZDOCK output file and return coordinates of poses or writes to PDB files
    
    Takes the ZDOCK output file, static and mobile PDB files as input
    Able to return the poses and write out PDB files
    """

    def __init__(self, zdock_output_file, zdock_static_file_path='', zdock_mobile_file_path=''):
        self.zdock_output_file = zdock_output_file
        self.zdock_static_file_path = zdock_static_file_path
        self.zdock_mobile_file_path = zdock_mobile_file_path
        self.grid_size = None
        self.grid_spacing = None
        self.switch = None
        self.recep_init_rot = None
        self.lig_init_rot = None
        self.recep_init_trans = None
        self.lig_init_trans = None
        self.num_poses = None
        self.zdock_output_data = None
        self.temp_static_selection_str = ''
        self.temp_mobile_selection_str = ''

        self.parse_zdock_output(self.zdock_output_file)
        self.static_uni = self.load_pdb_structures(self.process_ZDOCK_marked_file(self.zdock_static_file_path))
        self.mobile_uni = self.load_pdb_structures(self.process_ZDOCK_marked_file(self.zdock_mobile_file_path))

        if self.switch:
            self.reverse_init_lig_rot_mat = euler_to_rot_mat(-self.lig_init_rot[::-1])
            self.init_trans = self.lig_init_trans

        else:    
            self.reverse_init_recep_rot_mat = euler_to_rot_mat(-self.recep_init_rot[::-1])
            self.init_trans = self.recep_init_trans

        self.static_mobile_uni = MDAnalysis.Merge(self.static_uni.atoms, self.mobile_uni.atoms)
        self.static_mobile_copy_uni = MDAnalysis.Merge(self.static_uni.atoms, self.mobile_uni.atoms)

        self.static_mobile_copy_uni.trajectory.zdock_inst = self
        self.static_mobile_copy_uni.trajectory.n_frames = self.num_poses
        self.static_mobile_copy_uni.trajectory._read_next_timestep = types.MethodType(_read_next_timestep, self.static_mobile_copy_uni.trajectory)

        self.initial_mobile_coord = self.mobile_uni.atoms.positions
        self.initial_static_coord = self.static_uni.atoms.positions
        self.mobile_origin_coord = self.get_mobile_origin_coord(self.initial_mobile_coord)
        self.static_num_atoms = self.static_uni.atoms.n_atoms

    def get_mobile_origin_coord(self, coord):
        
        if self.switch:
            return self.apply_initial_rot_n_trans(self.recep_init_rot, self.recep_init_trans, coord)

        else:
            return self.apply_initial_rot_n_trans(self.lig_init_rot, self.lig_init_trans, coord)

    def process_ZDOCK_marked_file(self, marked_filename):
        new_pdb_str = ''
        pdb_file_lines = []

        if isinstance(marked_filename, cStringIO.OutputType):
            pdb_file_lines = marked_filename.readlines()

        else:

            with open(marked_filename, 'r') as marked_file:
                pdb_file_lines = marked_file.readlines()

        for line in pdb_file_lines:

            if line[0:6] in ['ATOM  ', 'HETATM']:
                new_pdb_str += line[0:54] + '\n'

        return common.convert_str_to_StrIO(new_pdb_str)

    def load_pdb_structures(self, pdb_stringio):
        return MDAnalysis.Universe(MDAnalysis.lib.util.NamedStream(pdb_stringio, 'marked.pdb'))

    def get_trans_vect(self, trans_vect, grid_size, grid_spacing):
        half_grid_size = grid_size/2.0
        gte_half_grid_size = trans_vect >= half_grid_size
        trans_vect[gte_half_grid_size] = grid_size - trans_vect[gte_half_grid_size]
        trans_vect[~gte_half_grid_size] *= -1
        trans_vect = trans_vect * grid_spacing
        
        return trans_vect

    def zdock_trans_rot(self, grid_size, grid_spacing, init_trans, mobile_rot, mobile_trans, init_coord, switch=False):

        if switch:
            dock_coord = init_coord - self.get_trans_vect(mobile_trans, grid_size, grid_spacing)
            dock_coord = rot_with_euler_rot(-mobile_rot[::-1], dock_coord) + init_trans
            dock_coord = dock_coord.dot(self.reverse_init_lig_rot_mat.T)

        else:
            dock_coord = rot_with_euler_rot(mobile_rot, init_coord)
            dock_coord += self.get_trans_vect(mobile_trans, grid_size, grid_spacing) + init_trans
            dock_coord = dock_coord.dot(self.reverse_init_recep_rot_mat.T)

        return dock_coord

    def parse_zdock_output(self, zdock_output_file):

        if isinstance(zdock_output_file, cStringIO.OutputType):
            zdock_output_lines = zdock_output_file.readlines()

        else:
            with open(zdock_output_file, 'r') as zdock_output_file_obj:
                zdock_output_lines = zdock_output_file_obj.readlines()

        self.grid_size = np.float32(zdock_output_lines[0].split()[0])
        self.grid_spacing = np.float32(zdock_output_lines[0].split()[1])
        self.switch = np.bool(np.int32(zdock_output_lines[0].split()[2]))

        if self.zdock_static_file_path == '':
            
            if self.switch:
                self.zdock_static_file_path = zdock_output_lines[4].split()[0]
            else:
                self.zdock_static_file_path = zdock_output_lines[3].split()[0]

        if self.zdock_mobile_file_path == '':

            if self.switch:
                self.zdock_mobile_file_path = zdock_output_lines[3].split()[0]
            else:
                self.zdock_mobile_file_path = zdock_output_lines[4].split()[0]

        # Euler rotation angles in ZDOCK output file are in: Z2, X1, Z1.
        # We will reverse the order to: Z1, X1, Z2.
        self.recep_init_rot = np.array(zdock_output_lines[1].split()[::-1], dtype='float32')
        self.lig_init_rot = np.array(zdock_output_lines[2].split()[::-1], dtype='float32')

        self.recep_init_trans = np.array(zdock_output_lines[3].split()[1:], dtype='float32')
        self.lig_init_trans = np.array(zdock_output_lines[4].split()[1:], dtype='float32')    
        
        self.num_poses = len(zdock_output_lines[5:])
        self.zdock_output_data = np.zeros((self.num_poses,7), dtype='float32')

        for idx, trans_rot_data in enumerate(zdock_output_lines[5:]):
            self.zdock_output_data[idx,:] = np.array(trans_rot_data.split(), dtype='float32')
            self.zdock_output_data[idx,:3] = self.zdock_output_data[idx,2::-1] # Reverse the order of the Euler angles

    def get_num_poses(self):
        return self.num_poses

    def set_mobile_selection(self, selection_str):

        if (selection_str != self.temp_mobile_selection_str):
            self.temp_mobile_selection = self.mobile_uni.select_atoms(selection_str)
            self.temp_mobile_selection_str = selection_str
    
    def set_static_selection(self, selection_str):

        if (selection_str != self.temp_static_selection_str):
            self.temp_static_selection = self.static_uni.select_atoms(selection_str)
            self.temp_static_selection_str = selection_str

    def apply_initial_rot_n_trans(self, initial_rot, initial_trans, coord):
        return rot_with_euler_rot(initial_rot, coord) - initial_trans

    def _set_pose_num(self, pose_num):
        """WARNING: Applies only to the MDAnalysis_Wrapper

        """
        current_mobile_coord = self.mobile_origin_coord

        self.static_mobile_copy_uni.trajectory.ts._pos[self.static_num_atoms:,:] = self.zdock_trans_rot(self.grid_size, self.grid_spacing, self.init_trans, self.zdock_output_data[pose_num-1,0:3].copy(), self.zdock_output_data[pose_num-1,3:6].copy(), current_mobile_coord, self.switch)

    def get_MDAnalysis_Wrapper(self):
        return self.static_mobile_copy_uni

    def get_pose(self, pose_num, mobile_only=False, static_sel_str=None, mobile_sel_str=None):

        if pose_num > self.num_poses:
            raise Exception('Pose number: %d is larger than total number of of poses: %d' % (pose_num, self.num_poses))

        current_mobile_coord = self.mobile_origin_coord
        current_static_coord = self.initial_static_coord

        if static_sel_str != None:
            self.set_static_selection(static_sel_str)
            current_static_coord = self.temp_static_selection.positions

        if mobile_sel_str != None:
            self.set_mobile_selection(mobile_sel_str)
            current_mobile_coord = self.get_mobile_origin_coord(self.temp_mobile_selection.positions)

        if mobile_only:
            return self.zdock_trans_rot(self.grid_size, self.grid_spacing, self.init_trans, self.zdock_output_data[pose_num-1,0:3].copy(), self.zdock_output_data[pose_num-1,3:6].copy(), current_mobile_coord, self.switch)

        else:
            return np.append(current_static_coord, self.zdock_trans_rot(self.grid_size, self.grid_spacing, self.init_trans, self.zdock_output_data[pose_num-1,0:3].copy(), self.zdock_output_data[pose_num-1,3:6].copy(), current_mobile_coord, self.switch), axis=0)

    def write_pose(self, pose_num, output_file_path, mobile_only=False):
        temp_coord = self.get_pose(pose_num, mobile_only=mobile_only)

        common.makedir(output_file_path)

        if mobile_only:
            self.mobile_uni.atoms.positions = temp_coord
            self.mobile_uni.atoms.write(output_file_path)
            self.mobile_uni.atoms.positions = self.initial_mobile_coord

        else:
            self.static_mobile_uni.atoms.positions = temp_coord
            self.static_mobile_uni.atoms.write(output_file_path)
