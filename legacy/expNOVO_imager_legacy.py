# -*- coding: UTF-8 -*-
'''
Created 24 November 2023
Hunter Ratliff

When ran on command line, use the -h or --help flags to see what options should be inputted.



# Quick search - search for one of the below strings to quickly get to that part of the code:
    - start_of_settings_section > file and image default settings
    - fastmode_filters > filters used to disallow events from fast imaging
    - build_neutron_cone_function > neutron cone building function
    - build_gamma_cone_function > gamma cone building function
    - start_of_imaging_section > image generation from cone projection section





This code seeks to take in sorted "events" and output reconstructed images.
Initially, the code was taking in PHITS data, but now it's agnostic to experimental vs MC data.
This script specifically is designed to be able to run standalone to turn a ROOT file of experimental data
into (1) a quickly generated image using filtered data and (2) images for a broader collection of individual
events to later be filterd and combined into new images in a separate code.
This code is intended to be used during experiments as a way to quickly get an initial image and then
prepare the data for more thorough image analysis.

The input is a list of events (as a pickle file) where each event is either
a) a neutron event with scatters in two bars, with the following information
   - 1st scatter : (x1,y1,z1), dE1 (proton energy deposited), t1 (timestamp 1)
   - 2nd scatter : (x2,y2,z2), t2 (timestamp 2)
b) a gamma event with scatters in three bars, with the following information
   - 1st scatter : (x1,y1,z1), dE1 (electron energy deposited)
   - 2nd scatter : (x2,y2,z2), dE2 (electron energy deposited)
   - 3rd scatter : (x3,y3,z3)

For neutrons, the positions and times of the two scatters allows a time-of-flight calculation to determine
En', the neutron energy after the first scatter. The cone angle is then theta = arcsin(sqrt(Ep/En')). This cone
has its axis on the line connecting the two scatters with its apex at the coordinates of the first scatter.

For gamma rays, one needs to find the initial gamma energy Eg and its energy following the first scatter, Eg'.
This requires first finding the scattering angle of the second scatter, theta2, which is found by taking the angle
between the vectors s1s2 and s2s3. Then, with theta2 and dE1 and dE2, Eg can be calculated via Eq 3 in the NOVO
feasibility study, linked below. Eg' = Eg - dE1 to get the gamma energy after the first scatter, then the initial
scattering angle is found as theta1 = arccos(1 + (me*c^2)[(1/Eg) - (1/Eg')]), with the cone constructed in the same
way as it is for the neutrons from the first two scatter coordinates and this angle.
Methods section in the feasibility study: https://www.nature.com/articles/s41598-023-33777-w#Sec7

The first part of this code serves to convert the input event data into event cones.

The input pickle file is a dictionary whose two main keys are 'neutron_records' and 'gamma_records' (along with other keys with metadata)
Each main key points to a numpy array using particle-specific dtypes shown below. The array's length is the number of
neutron/gamma events it contains. The bottom line of each type below contains information only used for Monte Carlo data.

neutron_event_record_type = np.dtype([('type', 'S1'),
                                 ('x1', np.single),('y1', np.single),('z1', np.single),('t1', np.single),('dE1', np.single),
                                 ('x2', np.single),('y2', np.single),('z2', np.single),('t2', np.single),
                                 ('protons_only',np.bool_),('theta_MCtruth', np.single)])
gamma_event_record_type   = np.dtype([('type', 'S1'),
                                 ('x1', np.single),('y1', np.single),('z1', np.single),('dE1', np.single),
                                 ('x2', np.single),('y2', np.single),('z2', np.single),('dE2', np.single),
                                 ('x3', np.single),('y3', np.single),('z3', np.single),('dE3', np.single),
                                 ('t1', np.single),('t2', np.single),('t3', np.single),('theta1_MCtruth', np.single)])




The second part of this code then creates an image using simple back projection (SBP) with these cones.

'''

import numpy as np
import os
import sys
import concurrent.futures
import multiprocessing as mp
import pickle
import time

import argparse
from pathlib import Path

def make_pickle(path_to_pickle_file, object_to_be_pickled):
    with open(path_to_pickle_file, 'wb') as handle:
        pickle.dump(object_to_be_pickled, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Pickle file written:', path_to_pickle_file, '\n')
    return


def read_pickle(path_to_pickle_file):
    with open(path_to_pickle_file, 'rb') as handle:
        extracted_data_from_pickle = pickle.load(handle)
    return extracted_data_from_pickle


def magnitude(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, degrees=False):
    """ Returns the angle in radians between vectors 'v1' and 'v2':: """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ang = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if degrees: ang = ang * 180 / np.pi
    return ang

print_debug_statements = False #True
print_debug_statements = True

if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter
    import re

    global output_filedir

    use_local_Hunters_tools = True
    if not use_local_Hunters_tools:
        from Hunters_tools import *

    from munch import *

    from collections import namedtuple
    import uproot

    use_fastmode = False
    use_CLI = False
    if len(sys.argv) > 1:
        use_CLI = True

        def dir_path(string):
            if os.path.isdir(string):
                return string
            else:
                raise NotADirectoryError(string)

        def file_path(string):
            if os.path.isfile(string):
                return string
            else:
                raise FileNotFoundError(string)


        parser = argparse.ArgumentParser()
        parser.add_argument("input")
        parser.add_argument("-f", "--fastmode", action="store_true",help='If used, code runs with aggressive filters (Elong>1/3 MeVee in scatter 1/2) and makes a fast, low-resolution image, not saving any event data for filtered events')
        parser.add_argument('-o','--outputdir', type=dir_path,nargs=1,help='Provide a separate directory into which imaging-related output files are stored. Default is <same directory as input file>/imaging')
        parser.add_argument('-p','--numprocesses', type=int,nargs=1,help='Specify number of processes to use in parallelized imaging. 0=Single process, Default=None=use all available processes.')
        parser.add_argument('-s','--settingsfile', type=file_path,nargs=1,help='Path to imaging settings file to overwrite default values in this code. Default is (in same directory as input file) "image_settings.txt"')


        args = parser.parse_args()

        input_path = Path(args.input)
        use_fastmode = args.fastmode
        num_processes = args.numprocesses
        if args.numprocesses!=None:
            num_processes=args.numprocesses[0]


        if use_fastmode:
            print('Creating aggressively cut, low-res, fast image')
        else:
            print('Creating comprehensive list-mode image to be further filtered later')

        if not input_path.exists():
            print("The target input file doesn't exist: ",input_path)
            raise SystemExit(1)

        if "fastmode" in str(input_path):
            use_fastmode = True

        input_path = input_path.resolve() # get full path (not just relative)
        input_filename = input_path.name
        input_filedir = input_path.parent # this is the path to the directory containing the supplied file

        if args.outputdir!=None:
            outputdir = args.outputdir[0]
            output_filedir = Path(outputdir).resolve()
        else:
            output_filedir = Path.joinpath(input_filedir, "imaging")
        if not output_filedir.exists():
            Path.mkdir(output_filedir)

        if args.settingsfile!=None:
            image_settings_file = str(Path(args.settingsfile[0]).resolve())
            image_settings_filename = Path(args.settingsfile[0]).resolve().name
        else:
            image_settings_file = str(input_filedir) + '/image_settings.txt'
            image_settings_filename = 'image_settings.txt'

        #print(input_path)
        #print(input_file)
        #print(input_filedir)

    else: # not CLI execution
        pass # defined later
        #image_settings_file = str(input_filedir) + '/image_settings.txt'
        #image_settings_filename = 'image_settings.txt'


    image_extensions = ['.pdf', '.png']

    show_plots = False  # True
    save_plots = True



    if use_CLI: show_plots = False

    neutron_event_record_type = np.dtype([('type', 'S1'),
                                          ('x1', np.single), ('y1', np.single), ('z1', np.single), ('t1', np.float64),
                                          ('dE1', np.single),
                                          ('x2', np.single), ('y2', np.single), ('z2', np.single), ('t2', np.float64),
                                          ('det1', np.short), ('det2', np.short), ('psd1', np.single),
                                          ('psd2', np.single), ('Elong1', np.single), ('Elong2', np.single),
                                          ('protons_only', np.bool_),
                                          ('theta_MCtruth',
                                           np.single)])  # ,#('xdir_MCtruth', np.single),('ydir_MCtruth', np.single),('zdir_MCtruth', np.single)])
    gamma_event_record_type = np.dtype([('type', 'S1'),
                                        ('x1', np.single), ('y1', np.single), ('z1', np.single), ('dE1', np.single), ('t1', np.float64),
                                        ('x2', np.single), ('y2', np.single), ('z2', np.single), ('dE2', np.single), ('t2', np.float64),
                                        ('x3', np.single), ('y3', np.single), ('z3', np.single), ('dE3', np.single), ('t3', np.float64),
                                        ('psd1', np.single), ('psd2', np.single), ('psd3', np.single),
                                        ('Elong1', np.single), ('Elong2', np.single), ('Elong3', np.single),
                                        ('det1', np.short), ('det2', np.short), ('det3', np.short),
                                        ('theta1_MCtruth',np.single), ('theta2_MCtruth',np.single)])

    # special record format just for passing to list mode imager
    neutron_event_record_type_condensed = np.dtype([('type', 'S1'),
                                                    ('tof', np.single), ('dE1', np.single),
                                                    ('det1', np.short), ('det2', np.short),
                                                    ('psd1', np.single), ('psd2', np.single),
                                                    ('Elong1', np.single), ('Elong2', np.single),
                                                    ])


    def NOVO_root2numpy(path_to_file_to_convert, syntax_style='Joey', start_par_is_gamma=True, bar1_IDs=[],
                        bar2_IDs=[], bar3_IDs=[]):
        global source_coordinates, save_plots, images_path, neutron_event_record_type, gamma_event_record_type, use_fastmode
        global output_filedir
        '''
        Converts ROOT data file from NOVO sort code (or simulation results emulating that output)
        to a pickled Numpy array.

        For Cf-252 and DT experiments, start detector timing is used along with scattering times
        to determine incident neutron energy, and thus dE1, instead of using true deposited energy since
        that energy calibration for neutrons hasn't been done yet.
        if start_par_is_gamma = True, the flight velocity between start detector and source is c
        if start_par_is_gamma = False, the flight velocity between start detector and source is that of a 3.6 MeV alpha

        bar1_IDs and bar2_IDs allows the code to filter for only events where the first and second interaction occur in the 
        specified bars.  If an empty list [], all bars are allowed for that scatter.
        
        max_n_entries_to_process = None processes all provided events, otherwise the code only processes the first of this number of events
        '''
        # save_plots = True
        # file_to_convert = imaging_code_testing_path + r'test_files\\true_NCE.root'

        if not use_CLI:
            inputfile_head, inputfile_tail = os.path.split(input_filepath)
            output_filedir = inputfile_head + '\\'

        if use_fastmode:
            imaging_record_data_pickle_path = Path.joinpath(output_filedir,os.path.basename(path_to_file_to_convert).replace('.root', '_fastmode.pickle'))
            #if use_CLI:
            #    imaging_record_data_pickle_path = Path.joinpath(output_filedir,os.path.basename(path_to_file_to_convert).replace('.root', '_fastmode.pickle'))
            #else:
            #    imaging_record_data_pickle_path = path_to_file_to_convert.replace('.root', '_fastmode.pickle')
        else:
            imaging_record_data_pickle_path = Path.joinpath(output_filedir,os.path.basename(path_to_file_to_convert).replace('.root', '.pickle'))
            #if use_CLI:
            #    imaging_record_data_pickle_path = Path.joinpath(output_filedir,os.path.basename(path_to_file_to_convert).replace('.root', '.pickle'))
            #else:
            #    imaging_record_data_pickle_path = path_to_file_to_convert.replace('.root', '.pickle')
        print('\tConverting file from ROOT to pickled Numpy array: ', path_to_file_to_convert)
        with uproot.open(path_to_file_to_convert) as file:
            # pass
            if print_debug_statements: print(file.keys())

            if syntax_style == 'Joey':
                key_tree_name = 'image_tree;1'
            elif syntax_style == 'Lena':
                key_tree_name = 'ntuple_NCE_raw;1'
            else:
                print('UNKNOWN VALUE PROVIDED FOR syntax_style IN FUNCTION NOVO_root2numpy')
                sys.exit()
            print_keys = False
            try:
                if print_debug_statements:
                    if print_keys: print(file[key_tree_name].keys())
                    try:
                        if print_keys: print('meta;1 keys:',file['meta;1'].keys())
                    except:
                        pass
                else:
                    tmp = (file[key_tree_name].keys())
            except:
                print('COULD NOT FIND DEFAULT IMAGE KEY, HERE ARE AVAILABLE KEYS')
                print(file.keys())
                print('USING: ', file.keys()[0])
                if syntax_style == 'Joey':
                    key_tree_name = file.keys()[0]

            if print_debug_statements: print(file[key_tree_name].keys())

            global ignore_settings_file, skip_gamma_events, start_det_to_hit1_max_gtime_ns, start_det_to_hit1_max_ntime_ns, min_allowed_tof12_ns
            global min_allowed_gamma_tof12_ns, max_allowed_tof12_ns, max_allowed_calc_E0, fast_n_min_elong1, fast_n_min_elong2, fast_g_min_elong
            global source_monoEn_MeV
            global fast_max_num_imaged_cones
            global n_min_elong1, n_min_elong2, g_min_elong, source_type
            global source_coordinates, NOVO_face_x_coord, axis_normal_to_array_face
            global max_n_entries_to_process

            skip_gamma_events = True  # if True, events with gamma timing are skipped
            start_det_to_hit1_max_gtime_ns = 15.0  # if abs(tstart-thit1)<this, assume incident par is gamma
            start_det_to_hit1_max_ntime_ns = 150.0 # if abs(tstart-thit1)<this, assume incident par is ultra slow neutron and skip it
            min_allowed_tof12_ns = 2.0 # reject events with shorter tof values (only if skip_gamma_events = True)
            min_allowed_gamma_tof12_ns = 0.5 # reject events with shorter tof values (only if skip_gamma_events = False)
            max_allowed_tof12_ns = 25.0 #9.0 # reject events with longer tof values
            max_allowed_calc_E0 = 15.0 # reject events whose neutron ToF energy > this (in MeV)
            source_monoEn_MeV = 14.1

            # fastmode_filters
            fast_n_min_elong1 = 3.0 # MeVee
            fast_n_min_elong2 = 1.0 # MeVee
            fast_g_min_elong  = 0.5 # MeVee

            # normal mode filter
            n_min_elong1 = 0.5 # MeVee
            n_min_elong2 = 0.5 # MeVee
            g_min_elong  = 0.2 # MeVee

            source_type = 'Cf252' # choose between 'Cf252', 'nELBE', and 'DT' ; this controls how En0 is determined

            if os.path.isfile(image_settings_file):
                print(f"Overwriting default event filtering criteria with those in {image_settings_filename}")
                f = open(image_settings_file)
                ignore_settings_file = False
                for line in f:
                    if ignore_settings_file: break
                    exec(line)
                f.close()


            if syntax_style == 'Joey':
                mult_to_convert_xyz_to_cm = 0.1
                calc_ndE1_from_dtof = True # currently does nothing
                start_detector_is_available = True # if false, a constant source energy is assumed
                #start_detector_is_available = False
                start_signal_is_shifted_tof_gflash_peak = False # for nELBE accelerator start signal
                calc_ndE1_from_DT = False # currently does nothing
                print('Found that "source_type" is set to: ',source_type)
                if source_type=='nELBE':
                    start_signal_is_shifted_tof_gflash_peak = True # for nELBE accelerator start signal
                elif source_type=='Cf252':
                    start_signal_is_shifted_tof_gflash_peak = False #True # for nELBE accelerator start signal
                elif source_type=='DT':
                    start_detector_is_available = False
                    calc_ndE1_from_DT = True
                else:
                    print("source_type not given a valid value; choose from:'Cf252', 'nELBE', and 'DT'")
                    sys.exit()

                # Check actual availability of start detector
                meta_keys = file['meta;1'].keys()
                start_detector_list = []
                for key in meta_keys:
                    if 'IsStartDet' in key:
                        if file['meta;1'][key].array()[0]==1: # 'Det14_IsStartDet'
                            start_detector_list.append(key.split('_')[0][3:])
                if start_detector_is_available and len(start_detector_list)==0:
                    print('WARNING: Start detector specified as available, but none listed in metadata')
                elif not start_detector_is_available and len(start_detector_list)>0:
                    print('WARNING: Start detector specified as NOT available, but the following are listed in metadata:')
                    print('\t',start_detector_list)
                if len(start_detector_list)>0:
                    start_det_coords = []
                    for isd in start_detector_list:
                        start_det_coords.append([ float(file['meta;1']['Det'+isd+'_PosX'].array()[0]),
                                                  float(file['meta;1']['Det'+isd+'_PosY'].array()[0]),
                                                  float(file['meta;1']['Det'+isd+'_PosZ'].array()[0])])
                    print(start_det_coords)


                dE1_data_available = False # this is if dE1 is the ACTUAL ENERGY lost in first scatter (in MeV, not MeVee)

                key_x1 = 'x1'
                key_y1 = 'y1'
                key_z1 = 'z1'
                key_t1 = 't1'
                key_dE1 = 'dE1'
                key_det1 = 'det1'
                key_par1 = 'particle1' # 0=neutron, 1=gamma, 2=laser, 3=unknown
                key_x2 = 'x2'
                key_y2 = 'y2'
                key_z2 = 'z2'
                key_t2 = 't2'
                key_dE2 = 'dE2'
                key_det2 = 'det2'
                key_par2 = 'particle2'
                key_x3 = 'x3'
                key_y3 = 'y3'
                key_z3 = 'z3'
                key_t3 = 't3'
                key_dE3 = 'dE3'
                key_det3 = 'det3'
                key_par3 = 'particle3'

                key_psd1 = 'psd1'
                key_psd2 = 'psd2'
                key_psd3 = 'psd3'

                key_xst = 'hit_xst'
                key_yst = 'hit_yst'
                key_zst = 'hit_zst'
                key_tst = 't0'
                key_dEst = 'hit_edepst'

            elif syntax_style == 'Lena':
                mult_to_convert_xyz_to_cm = 0.1
                calc_ndE1_from_dtof = False

                key_x1 = 'hit_x1'
                key_y1 = 'hit_y1'
                key_z1 = 'hit_z1'
                key_t1 = 'hit_time1'
                key_dE1 = 'hit_edep1'
                key_x2 = 'hit_x2'
                key_y2 = 'hit_y2'
                key_z2 = 'hit_z2'
                key_t2 = 'hit_time2'
                key_dE2 = 'hit_edep2'
                key_x3 = 'hit_x3'
                key_y3 = 'hit_y3'
                key_z3 = 'hit_z3'
                key_t3 = 'hit_time3'
                key_dE3 = 'hit_edep3'

                key_xst = 'hit_xst'
                key_yst = 'hit_yst'
                key_zst = 'hit_zst'
                key_tst = 'hit_timest'
                key_dEst = 'hit_edepst'

            n_entries = len(file[key_tree_name][key_x1].array())


            n_im_recs = np.empty((n_entries), dtype=neutron_event_record_type)  # neutron imaging records
            i_nir = 0  # current index of neutron imaging records
            g_im_recs = np.empty((n_entries), dtype=gamma_event_record_type)  # gamma imaging records
            i_gir = 0  # current index of gamma imaging records

            arr_x1 = file[key_tree_name][key_x1].array()
            arr_y1 = file[key_tree_name][key_y1].array()
            arr_z1 = file[key_tree_name][key_z1].array()
            arr_t1 = file[key_tree_name][key_t1].array()
            arr_dE1 = file[key_tree_name][key_dE1].array()
            arr_det1 = file[key_tree_name][key_det1].array()
            arr_x2 = file[key_tree_name][key_x2].array()
            arr_y2 = file[key_tree_name][key_y2].array()
            arr_z2 = file[key_tree_name][key_z2].array()
            arr_t2 = file[key_tree_name][key_t2].array()
            arr_dE2 = file[key_tree_name][key_dE2].array()
            arr_det2 = file[key_tree_name][key_det2].array()

            arr_psd1 = file[key_tree_name][key_psd1].array()
            arr_psd2 = file[key_tree_name][key_psd2].array()

            arr_par1 = file[key_tree_name][key_par1].array()
            arr_par2 = file[key_tree_name][key_par2].array()

            third_hit_data_available = False
            if not skip_gamma_events: third_hit_data_available = True
            if third_hit_data_available:
                arr_x3 = file[key_tree_name][key_x3].array()
                arr_y3 = file[key_tree_name][key_y3].array()
                arr_z3 = file[key_tree_name][key_z3].array()
                arr_t3 = file[key_tree_name][key_t3].array()
                arr_dE3= file[key_tree_name][key_dE3].array()
                arr_psd3 = file[key_tree_name][key_psd3].array()
                arr_det3 = file[key_tree_name][key_det3].array()
                arr_par3 = file[key_tree_name][key_par3].array()
            '''
            arr_xst = file[key_tree_name][key_xst].array()
            arr_yst = file[key_tree_name][key_yst].array()
            arr_zst = file[key_tree_name][key_zst].array()
            arr_tst = file[key_tree_name][key_tst].array()
            arr_dEst= file[key_tree_name][key_dEst].array()
            '''
            if start_detector_is_available:
                try: # new, Jan 2024, start detector handling
                    arr_tst = file[key_tree_name][key_tst].array()
                    arr_xst = np.ones(np.shape(arr_tst))*start_det_coords[0][0]
                    arr_yst = np.ones(np.shape(arr_tst))*start_det_coords[0][1]
                    arr_zst = np.ones(np.shape(arr_tst))*start_det_coords[0][2]
                except: # legacy start detector handling
                    arr_xst = file[key_tree_name][key_x3].array()
                    arr_yst = file[key_tree_name][key_y3].array()
                    arr_zst = file[key_tree_name][key_z3].array()
                    arr_tst = file[key_tree_name][key_t3].array()
                    arr_dEst = file[key_tree_name][key_dE3].array()
                    arr_detst = file[key_tree_name][key_det3].array()

            # if syntax_style=='Joey':
            #    meta_tree_name = 'meta;1'
            #    print(file[meta_tree_name].keys())
            #    sys.exit()

            srcx, srcy, srcz = source_coordinates
            c_vac = 29.979245800  # cm/ns
            mn = 939.565  # MeV/c^2
            mp = 938.272  # MeV/c^2
            mC = (12 * 931.494) - (6 * 0.511)  # MeV/c^2
            ma = 3727.3794066  # MeV/c^2
            # start_par_is_gamma = True # if False, assume alpha

            make_debug_plots = True
            # make_debug_plots = False

            tof_list = []
            tof12_list = []
            tof12_list_ALL_NN = []
            dist12_list = []
            En0_list = []
            Enprime_list = []

            #max_n_entries_to_process = 100000

            print('Starting ROOT to NumPy conversion...    ({:0.2f} seconds elapsed)'.format(time.time() - start))
            print('\t',n_entries, 'entries found...')
            if max_n_entries_to_process != None and max_n_entries_to_process<n_entries:
                print('\tonly the first',max_n_entries_to_process, 'entries will be processed    ...')
                n_entries = max_n_entries_to_process

            ccc = mult_to_convert_xyz_to_cm

            num_events_not_skipped_immediately = 0

            id_par_neutron = 1
            id_par_gamma = 2

            num_double_n = 0
            num_triple_g = 0
            num_ggg_dt12_too_low = 0
            num_ggg_dt12_too_high = 0
            num_nn_dt12_too_low = 0
            num_nn_dt12_too_high = 0
            num_nn_En0_too_high = 0
            num_nn_skipped_by_redundant_Ecuts = 0
            num_nn_not_in_allowed_bar_sequences = 0
            do_2023_nELBE_debugging_stuff = True
            # counters for temporary filters
            if do_2023_nELBE_debugging_stuff:
                num_events_skipped_by_temp_filters = 0
                num_double_start_hits = 0
                num_tof12_too_high = 0
                num_tof_too_high = 0
                num_tof_too_low = 0
                num_tof12_too_low = 0
                num_upscattering_events = 0
                num_NaN_dE1_events = 0
                num_0nn_1ng_2gg_3otr = [0,0,0,0] # tally of double neutron, n+g, double gamma, or other combo
                num_pre_E_cuts, num_post_E_cuts = 0,0

            for i in range(n_entries):
                # this assumes start detector is triggered by a fission gamma
                tof12_abs = abs(arr_t1[i]-arr_t2[i])
                this_is_a_gamma_event = False

                # if i>50: continue
                '''
                Skip all events that aren't double neutron or triple gamma
                '''
                par1, par2 = arr_par1[i], arr_par2[i]
                # 0=unidentified (no PSD cuts), 1=neutron, 2=gamma, 3=laser
                double_neutron = False
                if par1 == id_par_neutron and par2 == id_par_neutron:
                    num_0nn_1ng_2gg_3otr[0] += 1
                    num_double_n += 1
                    double_neutron = True
                    if make_debug_plots:
                        tof12_list_ALL_NN.append(tof12_abs)
                elif (par1 == id_par_gamma and par2 == id_par_neutron) or (
                        par1 == id_par_neutron and par2 == id_par_gamma):
                    num_0nn_1ng_2gg_3otr[1] += 1
                elif par1 == id_par_gamma and par2 == id_par_gamma:
                    num_0nn_1ng_2gg_3otr[2] += 1
                else:
                    num_0nn_1ng_2gg_3otr[3] += 1

                if not double_neutron and skip_gamma_events: continue

                triple_gamma = False
                if not skip_gamma_events and not double_neutron:
                    par3 = arr_par3[i]
                    if par1 == id_par_gamma and par2 == id_par_gamma and par3 == id_par_gamma:
                        triple_gamma = True
                        num_triple_g += 1
                    if not triple_gamma: continue

                if do_2023_nELBE_debugging_stuff:
                    '''
                    Temporary filters for nELBE December 2023 debugging
                    '''
                    #print([arr_dEst[i],arr_det1[i],arr_det2[i]],[arr_tst[i]-arr_t1[i],arr_tst[i]-arr_t2[i]])
                    if 14 in [arr_det1[i],arr_det2[i]] or 15 in [arr_det1[i],arr_det2[i]]:
                        num_events_skipped_by_temp_filters += 1
                        num_double_start_hits += 1
                        continue
                    if tof12_abs > max_allowed_tof12_ns:
                        num_events_skipped_by_temp_filters += 1
                        num_tof12_too_high += 1
                        continue
                    if tof12_abs < min_allowed_gamma_tof12_ns:
                        num_events_skipped_by_temp_filters += 1
                        num_tof12_too_low += 1
                        continue
                    if start_detector_is_available:
                        if abs(arr_tst[i]-min(arr_t1[i],arr_t2[i]))>300: # tof way too long
                            num_events_skipped_by_temp_filters += 1
                            num_tof_too_high += 1
                            continue
                        if abs(arr_tst[i]-min(arr_t1[i],arr_t2[i]))<5: # tof too short to be neutrons
                            num_events_skipped_by_temp_filters += 1
                            num_tof_too_low += 1
                            continue


                    #print([arr_detst[i],arr_det1[i],arr_det2[i]],[arr_tst[i]-arr_t1[i],arr_tst[i]-arr_t2[i]])
                    '''
                    Notes for Joey:
                    - For vast majority of events, calcs give upscattering dE1, nonsensical
                    '''


                #double_neutron = False
                #if par1==id_par_neutron and par2==id_par_neutron:
                #    double_neutron = True

                #if tof12_abs > 100: continue # max allowed tof12
                #if tof12_abs < min_allowed_tof12_ns:  # gamma events fast filters  <--- old method for filtering gammas
                num_pre_E_cuts += 1
                if not double_neutron: # not a double neutron event
                    #print(arr_par1[i],arr_par2[i])
                    if skip_gamma_events: continue
                    if tof12_abs < min_allowed_gamma_tof12_ns: continue
                    this_is_a_gamma_event = True
                    if use_fastmode:
                        if arr_dE1[i]<fast_g_min_elong: continue
                        if arr_dE2[i]<fast_g_min_elong: continue
                        if arr_dE3[i]<fast_g_min_elong: continue
                    else:
                        if arr_dE1[i]<g_min_elong: continue
                        if arr_dE2[i]<g_min_elong: continue
                        if arr_dE3[i]<g_min_elong: continue
                if not this_is_a_gamma_event: # neutron events fast filters
                    if use_fastmode:
                        if arr_t1[i] < arr_t2[i]:
                            if arr_dE1[i]<fast_n_min_elong1: continue
                            if arr_dE2[i]<fast_n_min_elong2: continue
                        else:
                            if arr_dE2[i]<fast_n_min_elong1: continue
                            if arr_dE1[i]<fast_n_min_elong2: continue
                    else:
                        if arr_t1[i] < arr_t2[i]:
                            if arr_dE1[i]<n_min_elong1: continue
                            if arr_dE2[i]<n_min_elong2: continue
                        else:
                            if arr_dE2[i]<n_min_elong1: continue
                            if arr_dE1[i]<n_min_elong2: continue
                num_post_E_cuts += 1
                num_events_not_skipped_immediately += 1

                #if use_fastmode:
                #    if i_gir>2*fast_max_num_imaged_cones: continue
                #t_hit1, t_hit2, t_hit3 = 0,0,0
                if not third_hit_data_available or double_neutron:
                    if arr_t1[i] < arr_t2[i]:
                        s1, s2 = '1', '2'
                        t_hit1 = arr_t1[i]
                        t_hit2 = arr_t2[i]
                        x1, y1, z1 = arr_x1[i], arr_y1[i], arr_z1[i]
                        x2, y2, z2 = arr_x2[i], arr_y2[i], arr_z2[i]
                        det1, det2 = arr_det1[i], arr_det2[i]
                        elong1, elong2 = arr_dE1[i], arr_dE2[i]
                    else:
                        s1, s2 = '2', '1'
                        t_hit1 = arr_t2[i]
                        t_hit2 = arr_t1[i]
                        x1, y1, z1 = arr_x2[i], arr_y2[i], arr_z2[i]
                        x2, y2, z2 = arr_x1[i], arr_y1[i], arr_z1[i]
                        det1, det2 = arr_det2[i], arr_det1[i]
                        elong1, elong2 = arr_dE2[i], arr_dE1[i]
                else:
                    if arr_t1[i] < arr_t2[i] < arr_t3[i]:
                        s1, s2, s3 = '1', '2', '3'
                    elif arr_t1[i] < arr_t3[i] < arr_t2[i]:
                        s1, s2, s3 = '1', '3', '2'
                    elif arr_t2[i] < arr_t1[i] < arr_t3[i]:
                        s1, s2, s3 = '2', '1', '3'
                    elif arr_t2[i] < arr_t3[i] < arr_t1[i]:
                        s1, s2, s3 = '2', '3', '1'
                    elif arr_t3[i] < arr_t1[i] < arr_t2[i]:
                        s1, s2, s3 = '3', '1', '2'
                    elif arr_t3[i] < arr_t2[i] < arr_t1[i]:
                        s1, s2, s3 = '3', '2', '1'
                    det3, t_hit1, t_hit2, t_hit3 = None, None, None, None # placeholder to make editor happy (since defined only in exec() below)
                    elong3 = None
                    lcls = locals()
                    exec("t_hit1, t_hit2, t_hit3 = arr_t"+s1+"[i], arr_t"+s2+"[i], arr_t"+s3+"[i]", globals(), lcls)
                    exec("x1, y1, z1 = arr_x"+s1+"[i], arr_y"+s1+"[i], arr_z"+s1+"[i]", globals(), lcls)
                    exec("x2, y2, z2 = arr_x"+s2+"[i], arr_y"+s2+"[i], arr_z"+s2+"[i]", globals(), lcls)
                    exec("x3, y3, z3 = arr_x"+s3+"[i], arr_y"+s3+"[i], arr_z"+s3+"[i]", globals(), lcls)
                    exec("det1, det2, det3 = arr_det"+s1+"[i], arr_det"+s2+"[i], arr_det"+s3+"[i]", globals(), lcls)
                    exec("elong1, elong2, elong3 = arr_dE"+s1+"[i], arr_dE"+s2+"[i], arr_dE"+s3+"[i]", globals(), lcls)
                    t_hit1, t_hit2, t_hit3 = lcls['t_hit1'], lcls['t_hit2'], lcls['t_hit3']
                    x1, y1, z1 = lcls['x1'], lcls['y1'], lcls['z1']
                    x2, y2, z2 = lcls['x2'], lcls['y2'], lcls['z2']
                    x3, y3, z3 = lcls['x3'], lcls['y3'], lcls['z3']
                    det1, det2, det3 = lcls['det1'], lcls['det2'], lcls['det3']
                    elong1, elong2, elong3 = lcls['elong1'], lcls['elong2'], lcls['elong3']
                    #print(t_hit1, t_hit2, t_hit3, s1, s2, s3, arr_t1[i], arr_t2[i], arr_t3[i])
                    #print("t_hit1, t_hit2, t_hit3 = arr_t"+s1+"[i], arr_t"+s2+"[i], arr_t"+s3+"[i]")


                tof12 = t_hit2 - t_hit1
                tof12_abs = abs(tof12)

                #print('array time vals:',arr_t1[i],arr_t2[i],'\t\t t_hit vals:',t_hit1,t_hit2)
                #print('t_hit1=',t_hit1,'=',arr_t1[i],'ns     t_hit2=',t_hit2,'=',arr_t2[i],'ns')
                #if i>50: sys.exit()

                #if tof12_abs < min_allowed_tof12_ns:  # gammas, just pass along data
                if triple_gamma:
                    if skip_gamma_events:
                        #print(tof12_abs)
                        continue
                    if tof12_abs < min_allowed_gamma_tof12_ns:
                        num_ggg_dt12_too_low += 1
                        continue
                    if tof12_abs > max_allowed_tof12_ns:
                        num_ggg_dt12_too_high += 1
                    this_is_a_gamma_event = True

                    #if use_fastmode:
                    #    if arr_dE1[i]<fast_g_min_elong: continue
                    #    if arr_dE2[i]<fast_g_min_elong: continue
                    #    if arr_dE3[i]<fast_g_min_elong: continue

                    if len(bar1_IDs) > 0 or len(bar2_IDs) > 0 or len(bar3_IDs) > 0:  # requirements placed on both first and scatter bars
                        if len(bar1_IDs) > 0:  # requirement placed on where first scatter happened
                            if (det1 not in bar1_IDs):
                                continue
                        if len(bar2_IDs) > 0:  # requirement placed on where second scatter happened
                            if (det2 not in bar2_IDs):
                                continue
                        if len(bar3_IDs) > 0:  # requirement placed on where third scatter happened
                            if (det3 not in bar3_IDs):
                                continue

                    g_im_recs['type'][i_gir] = 'g'
                    g_im_recs['x' + s1][i_gir] = arr_x1[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['y' + s1][i_gir] = arr_y1[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['z' + s1][i_gir] = arr_z1[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['t' + s1][i_gir] = arr_t1[i]
                    g_im_recs['psd' + s1][i_gir] = arr_psd1[i]
                    g_im_recs['det' + s1][i_gir] = arr_det1[i]
                    g_im_recs['Elong' + s1][i_gir] = arr_dE1[i]
                    g_im_recs['dE'+s1][i_gir]= arr_dE1[i]
                    #g_im_recs['dE1'][i_gir] = dE1
                    g_im_recs['x' + s2][i_gir] = arr_x2[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['y' + s2][i_gir] = arr_y2[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['z' + s2][i_gir] = arr_z2[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['t' + s2][i_gir] = arr_t2[i]
                    g_im_recs['psd' + s2][i_gir] = arr_psd2[i]
                    g_im_recs['det' + s2][i_gir] = arr_det2[i]
                    g_im_recs['Elong' + s2][i_gir] = arr_dE2[i]
                    g_im_recs['dE'+s2][i_gir]= arr_dE2[i]
                    #g_im_recs['dE2'][i_gir] = dE2
                    g_im_recs['x' + s3][i_gir] = arr_x3[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['y' + s3][i_gir] = arr_y3[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['z' + s3][i_gir] = arr_z3[i] * mult_to_convert_xyz_to_cm
                    g_im_recs['t' + s3][i_gir] = arr_t3[i]
                    g_im_recs['psd' + s3][i_gir] = arr_psd3[i]
                    g_im_recs['det' + s3][i_gir] = arr_det3[i]
                    g_im_recs['Elong' + s3][i_gir] = arr_dE3[i]
                    g_im_recs['dE'+s3][i_gir]= arr_dE3[i]

                    i_gir += 1
                    continue
                # neutron events if reaching here
                if not double_neutron: continue

                #if use_fastmode:
                #    if arr_dE1[i]<fast_n_min_elong1: continue
                #    if arr_dE2[i]<fast_n_min_elong2: continue
                #    #if i_nir>2*fast_max_num_imaged_cones: continue

                if dE1_data_available:
                    if arr_t1[i] < arr_t2[i]:
                        dE1 = arr_dE1[i]
                    else:
                        dE1 = arr_dE2[i]

                else:  # must calculate dE1 by some means, namely by dE1 = En_prime - En0, need to find En0

                    if not start_detector_is_available:  # assume DT neutrons
                        En0 = source_monoEn_MeV

                    else:  # use start detector
                        # print(x1,y1,z1,t_hit1,x2,y2,z2,t_hit2)#,x3,y3,z3,t3)
                        # print(t_hit1-t_hit2)

                        if start_signal_is_shifted_tof_gflash_peak:
                            # at nELBE, accelerator start used to time calibrate tof spectra
                            # gamma peak is already shifted to correct tof from source location
                            # such that t_start is the "zero" time when the neutrons/gammas were spawned
                            t_strart = arr_tst[i]
                            tof = t_hit1 - t_strart
                            #tof = abs(tof)
                        else:

                            max_dt_ns = max_allowed_tof12_ns #20.0
                            if tof12_abs > max_dt_ns:
                                num_nn_dt12_too_high += 1
                                #print(tof12_abs , max_dt_ns)
                                continue

                            t_strart = arr_tst[i]
                            d_src2start = dist(source_coordinates, [arr_xst[i] * ccc, arr_yst[i] * ccc, arr_zst[i] * ccc])
                            #print(d_src2start,source_coordinates, [arr_xst[i] * ccc, arr_yst[i] * ccc, arr_zst[i] * ccc])

                            if start_par_is_gamma:
                                v_start_par_cmpns = c_vac  # cm/ns
                            else:  # assume alpha
                                # in principle this is in arr_dEst, but could be quenched
                                # therefore, just assume DT fusion alpha energy
                                E_alpha = 3.5  # MeV
                                v_start_par_cmpns = c_vac * np.sqrt((2 * E_alpha) / ma)
                            t_n_spawn = t_strart - (d_src2start / v_start_par_cmpns)
                            tof = t_hit1 - t_n_spawn  # cm

                        d_src2hit1 = dist(source_coordinates, [x1 * ccc, y1 * ccc, z1 * ccc])  # cm
                        #print(d_src2hit1 ,source_coordinates, [x1 * ccc, y1 * ccc, z1 * ccc])
                        v_n0 = d_src2hit1 / tof
                        L_gamma = Lorentz_gamma(v_n0)  # unitless
                        En0 = (L_gamma - 1) * mn  # MeV
                        # if np.isnan(En0): continue

                        #print(En0)

                    d_hit12hit2 = dist([x1 * ccc, y1 * ccc, z1 * ccc], [x2 * ccc, y2 * ccc, z2 * ccc])
                    #tof12 = t_hit2 - t_hit1
                    #print("En0={:g} MeV, d12={:g} cm, tof={:g} ns, tof12={:g} ns, hits in dets {:g} and {:g}".format(En0,d_hit12hit2,tof,tof12_abs,arr_det1[i],arr_det2[i]))

                    # in initial testing before any cuts are applied, manually filter out events with gamma TOFs
                    #this_is_a_gamma_event = False
                    if start_detector_is_available:
                        #print(abs(tof), start_det_to_hit1_max_gtime_ns)
                        if abs(tof) < start_det_to_hit1_max_gtime_ns:   # gammas
                            #print(tof,start_det_to_hit1_max_gtime_ns)
                            if skip_gamma_events: continue
                            if tof12_abs < min_allowed_gamma_tof12_ns: continue
                            this_is_a_gamma_event = True
                        if tof < 0:
                            #tof = abs(tof)
                            continue  # ??? start fires after bar, reject
                        if abs(tof) > start_det_to_hit1_max_ntime_ns: continue  # very slow neutrons

                    if tof12_abs < min_allowed_tof12_ns: # neutron flight time too short
                        num_nn_dt12_too_low += 1
                        continue # gammas should not get this far and be sorted earlier...
                        '''
                        if skip_gamma_events: continue
                        if tof12_abs < min_allowed_gamma_tof12_ns:
                            #num_ggg_dt12_too_low += 1
                            continue
                        this_is_a_gamma_event = True
                        '''
                    if tof12_abs > max_allowed_tof12_ns:
                        #print(tof12_abs)
                        num_nn_dt12_too_high += 1
                        continue  # trash
                    # if np.sqrt(arr_psd1[i]*arr_psd2[i])<0.375: continue # gammas
                    if En0 > max_allowed_calc_E0:
                        #print(En0)
                        num_nn_En0_too_high += 1
                        continue  # neutrons of E this high probably not physically from Cf252

                    # energy cuts
                    if arr_t1[i] < arr_t2[i]:
                        dE1_cut = arr_dE1[i]
                        dE2_cut = arr_dE2[i]
                        det1_cut = arr_det1[i]
                        det2_cut = arr_det2[i]
                    else:
                        dE1_cut = arr_dE2[i]
                        dE2_cut = arr_dE1[i]
                        det1_cut = arr_det2[i]
                        det2_cut = arr_det1[i]
                    # print(dE1_cut,dE2_cut)


                    # This should do nothing now that data should be coming in time ordered...
                    num_nn_skipped_by_redundant_Ecuts += 1
                    if use_fastmode:
                        if dE1_cut < fast_n_min_elong1: continue  # MeV
                        if dE2_cut < fast_n_min_elong2: continue  # MeV
                    else:
                        #print(dE1_cut,dE2_cut)
                        if dE1_cut < n_min_elong1: continue  # MeV
                        if dE2_cut < n_min_elong2: continue  # MeV
                    num_nn_skipped_by_redundant_Ecuts -= 1

                    #print('hi',dE1_cut,n_min_elong1,dE2_cut,n_min_elong2)

                    # image only from certain bars
                    # if not (det1_cut==3 and det2_cut==1): continue
                    # if not (det1_cut==3 and det2_cut==4): continue

                    # if not (det1_cut%2==0 and det2_cut%2==0): continue # if not OGS
                    # if not (det1_cut%2==1 and det2_cut%2==1): continue # if not Target M600
                    if len(bar1_IDs) > 0 or len(bar2_IDs) > 0:
                        num_nn_not_in_allowed_bar_sequences += 1
                        if len(bar1_IDs) > 0 and len(bar2_IDs) > 0:  # requirements placed on both first and scatter bars
                            if not (det1 in bar1_IDs and det2 in bar2_IDs):
                                continue
                        elif len(bar1_IDs) > 0:  # requirement only placed on where first scatter happened
                            if (det1 not in bar1_IDs):
                                continue
                        elif len(bar2_IDs) > 0:  # requirement only placed on where second scatter happened
                            if (det2 not in bar2_IDs):
                                continue
                        num_nn_not_in_allowed_bar_sequences -= 1
                    else:
                        pass

                    v_nprime = d_hit12hit2 / tof12
                    L_gamma_prime = Lorentz_gamma(v_nprime)  # unitless
                    Enprime = (L_gamma_prime - 1) * mn  # MeV
                    dE1 = En0 - Enprime
                    #print(En0,Enprime,v_nprime,d_hit12hit2,tof12)
                    if np.isnan(dE1):
                        if do_2023_nELBE_debugging_stuff: num_NaN_dE1_events += 1
                        continue
                    if dE1 < 0:
                        if do_2023_nELBE_debugging_stuff: num_upscattering_events += 1
                        continue
                    # if not start_detector_is_available:
                    #    if Enprime < 1.0: continue
                    '''
                    if do_2023_nELBE_debugging_stuff: # print info for surviving events
                        print("En0={:g} MeV, En'={:g} MeV, d12={:g} cm, tof={:g} ns, dist={:g} cm, tof12={:g} ns, hits in dets {:g} and {:g}".format(En0,Enprime,d_hit12hit2,tof,d_src2hit1,tof12_abs,arr_det1[i],arr_det2[i]))
                        #print('\t',use_fastmode,fast_n_min_elong1,fast_n_min_elong2, [x1 * ccc, y1 * ccc, z1 * ccc])
                        pass
                    '''
                if make_debug_plots:
                    En0_list.append(En0)
                    if start_detector_is_available:tof_list.append(tof)
                    Enprime_list.append(Enprime)
                    tof12_list.append(tof12)
                    dist12_list.append(d_hit12hit2)


                n_im_recs['type'][i_nir] = 'n'
                n_im_recs['x' + s1][i_nir] = arr_x1[i] * mult_to_convert_xyz_to_cm
                n_im_recs['y' + s1][i_nir] = arr_y1[i] * mult_to_convert_xyz_to_cm
                n_im_recs['z' + s1][i_nir] = arr_z1[i] * mult_to_convert_xyz_to_cm
                n_im_recs['t' + s1][i_nir] = arr_t1[i]
                n_im_recs['psd' + s1][i_nir] = arr_psd1[i]
                n_im_recs['det' + s1][i_nir] = arr_det1[i]
                n_im_recs['Elong' + s1][i_nir] = arr_dE1[i]
                # n_im_recs['dE'+s1][i_nir]= arr_dE1[i]
                n_im_recs['dE1'][i_nir] = dE1
                n_im_recs['x' + s2][i_nir] = arr_x2[i] * mult_to_convert_xyz_to_cm
                n_im_recs['y' + s2][i_nir] = arr_y2[i] * mult_to_convert_xyz_to_cm
                n_im_recs['z' + s2][i_nir] = arr_z2[i] * mult_to_convert_xyz_to_cm
                n_im_recs['t' + s2][i_nir] = arr_t2[i]
                n_im_recs['psd' + s2][i_nir] = arr_psd2[i]
                n_im_recs['det' + s2][i_nir] = arr_det2[i]
                n_im_recs['Elong' + s2][i_nir] = arr_dE2[i]
                ###n_im_recs['dE2'][i_nir]= arr_dE2[i]
                i_nir += 1

                # print(t_hit1, t_hit2, abs(t_hit1 - t_hit2))
                # print(n_im_recs['t'+s1][i_nir], n_im_recs['t'+s2][i_nir], abs(n_im_recs['t'+s2][i_nir] - n_im_recs['t'+s1][i_nir]))
                # sys.exit()

            if do_2023_nELBE_debugging_stuff:
                print('\tof which,',num_events_not_skipped_immediately ,'were not skipped immediately and')
                print('\tof which,',num_events_skipped_by_temp_filters, 'were skipped by temporary filters:')
                print('\t\t',num_double_start_hits,'events with 2 start detectors firing')
                print('\t\t',num_tof12_too_high,'events with ToF between bars too long')
                print('\t\t',num_tof12_too_low,'events with ToF between bars too short')
                print('\t\t',num_tof_too_low,'events with ToF between bar1hit and start detector too short')
                print('\t\t',num_tof_too_high,'events with ToF between bar1hit and start detector too high')
                print('\t',num_pre_E_cuts-num_post_E_cuts,'events failing Elong cuts')
                print('\t',num_upscattering_events,'events skipped due to upscattering (E1>E0)')
                print('\t',num_NaN_dE1_events,'events skipped due to NaN scattered neutron energy')
                print('\t',num_ggg_dt12_too_low,'triple g events skipped due to dt12 less than',min_allowed_gamma_tof12_ns,'ns')
                print('\t',num_ggg_dt12_too_high,'triple g events skipped due to dt12 greater than',max_allowed_tof12_ns,'ns')
                print('\t',num_nn_dt12_too_low,'double n events skipped due to dt12 less than',min_allowed_tof12_ns,'ns')
                print('\t',num_nn_dt12_too_high,'double n events skipped due to dt12 greater than',max_allowed_tof12_ns,'ns')
                print('\t',num_nn_En0_too_high,'double n events skipped due to En0 greater than',max_allowed_calc_E0,'MeV')
                print('\t',num_nn_skipped_by_redundant_Ecuts,'double n events skipped by "redundant" energy cuts (should be 0)')
                print('\t',num_nn_not_in_allowed_bar_sequences,'double n events skipped due to not matching inputted required bar sequences')
                print('\n\t',i_nir,'neutron events will be imaged, of',num_double_n,'double n events initially found')
                print('\n\t',i_gir,  'gamma events will be imaged, of',num_triple_g,'triple g events initially found')
                print('\n','Of {:g} events: {:g} double n, {:g} n+g, {:g} double g, {:g} other combo'.format(sum(num_0nn_1ng_2gg_3otr),num_0nn_1ng_2gg_3otr[0],num_0nn_1ng_2gg_3otr[1],num_0nn_1ng_2gg_3otr[2],num_0nn_1ng_2gg_3otr[3]))

                #sys.exit()


            n_im_recs = n_im_recs[:i_nir]
            g_im_recs = g_im_recs[:i_gir]


            print('ROOT to NumPy conversion complete!    ({:0.2f} seconds elapsed)'.format(time.time() - start))

            if i_nir==0 and i_gir==0:
                print('NO EVENTS PASSED CONVERSION CHECKS, ABORTING!!! Pickle file NOT written.')
                sys.exit()

            print(i_gir,'gamma cone candidates found,',i_nir,'neutron cone candidates found')
            if i_gir==0: sys.exit()


            if make_debug_plots:
                if start_detector_is_available:
                    En_range = None  # [0,10]
                    plt.figure()
                    plt.hist(En0_list, bins='auto', label='En0', alpha=0.5, range=En_range)
                    plt.xlabel('En [MeV]')
                    plt.legend()

                    tof_range = None  # [0,10]
                    plt.figure()
                    plt.hist(tof_list, bins='auto', label='ToF, start-hit1', alpha=0.5, range=tof_range)
                    plt.xlabel('ToF [ns]')
                    plt.legend()

                En_range = None  # [0,10]
                figd1 = plt.figure()
                plt.hist(Enprime_list, bins='auto', label='En,prime', alpha=0.5, range=En_range)
                plt.xlabel('En [MeV]')
                plt.legend()
                if save_plots:
                    for ext in image_extensions:
                        plot_save_path = images_path + 'debug_En-prime' + ext  # or use fig.canvas.get_window_title()
                        figd1.savefig(plot_save_path, facecolor=(0, 0, 0, 0))

                tof_range = None # [0, 10]
                figd2 = plt.figure()
                plt.hist(tof12_list, bins=100, label='ToF, hit1-hit2', alpha=0.5)#, range=tof_range)
                plt.xlabel('ToF [ns] of imaged double neutrons')
                plt.legend()
                print('debug ToF info for imaged double neutron events: min ToF =,',min(tof12_list),'ns, max ToF =',max(tof12_list),' ns\n')
                if save_plots:
                    for ext in image_extensions:
                        plot_save_path = images_path + 'debug_ToF' + ext  # or use fig.canvas.get_window_title()
                        figd2.savefig(plot_save_path, facecolor=(0, 0, 0, 0))

                tof_range = None  # [0, 10]
                figd2b = plt.figure()
                plt.hist(tof12_list_ALL_NN, bins=100, label='ToF, hit1-hit2', alpha=0.5)  # , range=tof_range)
                plt.xlabel('ToF [ns] of ALL double neutron events')
                plt.legend()
                print('debug ToF info for ALL double neutron events: min ToF =,', min(tof12_list_ALL_NN), 'ns, max ToF =', max(tof12_list_ALL_NN), ' ns\n')
                if save_plots:
                    for ext in image_extensions:
                        plot_save_path = images_path + 'debug_ToF_ALL_NN' + ext  # or use fig.canvas.get_window_title()
                        figd2b.savefig(plot_save_path, facecolor=(0, 0, 0, 0))

                dist12_range = None # [6, 20]
                figd3 = plt.figure()
                plt.hist(dist12_list, bins='auto', label='dist, hit1-hit2', alpha=0.5, range=dist12_range)
                plt.xlabel('distance [cm]')
                plt.legend()
                if save_plots:
                    for ext in image_extensions:
                        plot_save_path = images_path + 'debug_dists' + ext  # or use fig.canvas.get_window_title()
                        figd3.savefig(plot_save_path, facecolor=(0, 0, 0, 0))

                # plt.show()

            with open(imaging_record_data_pickle_path, 'wb') as handle:
                to_be_pickled = {'neutron_records': n_im_recs, 'gamma_records': g_im_recs,
                                 'sim_base_folder_name': path_to_file_to_convert}
                pickle.dump(Munch(to_be_pickled), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('\tPickle file written:', imaging_record_data_pickle_path, '\n')

            print('\t Recorded {:g} neutron events and {:g} gamma events for potential imaging.'.format(i_nir, i_gir))
        return





    if use_local_Hunters_tools: # BIG IMPORT OF Hunters_tools.py module stuff
        import unicodedata as ud
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.ticker as mticker
        from matplotlib import cm
        from mpl_toolkits.mplot3d.axis3d import Axis
        import matplotlib.projections as proj
        from matplotlib.colors import colorConverter, LinearSegmentedColormap
        import matplotlib.ticker as ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy.interpolate import CubicSpline, lagrange, interp1d
        from lmfit.models import GaussianModel
        from lmfit import Parameter
        from scipy.optimize import curve_fit
        from scipy import stats
        from scipy.stats import chisquare

        # extra distributions
        #from lmfit.models import LorentzianModel

        def Lorentz_gamma(v):
            '''
            Description:
                Calculate the Lorentz variable gamma provided a velocity in cm/ns
            Inputs:
              - `v` = velocity in cm/ns (`float` ot `int`)
            Outputs:
              - `gamma` = Lorentz variable gamma
            '''
            m_n = 939.5654133  # neutron mass in MeV/c^2
            c_vac = 29979245800  # cm/s
            refractive_index_air = 1.0003
            c = c_vac / refractive_index_air
            c_cm_per_ns = c / (10 ** 9)
            beta = v / c_cm_per_ns
            gamma = np.sqrt(1 / (1 - beta ** 2))
            return gamma


        def Lorentz_B2_from_Tn(Tn):
            '''
            Description:
                Calculate the Lorentz variable beta^2 provided a neutron kinetic energy in MeV
            Inputs:
              - `Tn` = neutron kinetic energy in MeV (`float` ot `int`)
            Outputs:
              - `beta_squared` = Lorentz variable beta^2
            '''
            m_n = 939.5654133  # neutron mass in MeV/c^2
            c_vac = 29979245800  # cm/s
            refractive_index_air = 1.0003
            c = c_vac / refractive_index_air
            c_cm_per_ns = c / (10 ** 9)
            gamma = 1 + (Tn / m_n)
            beta_squared = 1 - (1 / (gamma ** 2))
            return beta_squared


        def dist(a, b):
            '''
            Description:
                Calculate the distance between two N-dimensional Cartesian points a and b
            Dependencies:
                `import numpy as np`
            Inputs:
              - `a` = length N list containing coordinates of the first point, a  (ex. [ax,ay] or [ax,ay,az])
              - `b` = length N list containing coordinates of the second point, b (same format as a)
            Outputs:
              - `d` = Cartesian distance between points a and b

            '''
            if len(a) != len(b):
                print('a ({}) and b ({}) are not of the same dimension'.format(str(len(a)), str(len(b))))
                return 0
            d = 0
            for i in range(len(a)):
                d += (a[i] - b[i]) ** 2
            d = np.sqrt(d)
            return d


        def slugify(value):
            '''
            Description:
                Normalizes string, converts to lowercase, removes non-alpha characters,and converts spaces to hyphens.
                This is useful for quickly converting TeX strings, plot titles, etc. into legal filenames.

            Dependencies:
                - `import unicodedata as ud`
                - `import re`

            Input:
                - `value` = string to be "slugified"

            Output:
                - `value` converted to a string only consisting of characters legal in filenames
            '''
            old_value = value
            value = str(ud.normalize('NFKD', value).encode('ascii', 'ignore'))
            value = str(re.sub('[^\w\s-]', '', value).strip().lower())
            value = str(re.sub('[-\s]+', '-', value))
            if value[0] == 'b' and old_value[0] != 'b': value = value[
                                                                1:]  # TeX strings sometimes case resulting string to being with 'b'
            return value


        def makeErrorBoxes(ax, xdata, ydata, xerror, yerror, fc='None', ec='k', alpha=1.0, lw=0.5):
            '''
            Description:
                Generate uncertainty/error "boxes" which are overlaid on points

            Dependencies:
              - `import numpy as np`
              - `import matplotlib.pyplot as plt`
              - `from matplotlib.collections import PatchCollection`
              - `from matplotlib.patches import Rectangle`

            Inputs:
               (required)

              - `ax` = axis handles onto which error boxes will be drawn
              - `xdata` = a list/array of x data
              - `ydata` = a list/array of y data
              - `xerror` = a list/array of 2 lists/arrays of x absolute uncertainties as [x_lower_errors, x_upper_errors]
              - `yerror` = a list/array of 2 lists/arrays of y absolute uncertainties as [y_lower_errors, y_upper_errors]

            Inputs:
               (optional)

              - `fc` = face color of boxes (D=`None`)
              - `ec` = edge color of boxes (D=`'k'`, black)
              - `alpha` = opacity of box filling (D=`1.0`)
              - `lw` = line width of box edge (D=`0.5`)

            Notes:
                For best results, repeat this function twice, first rendering the edges and then a second time for the filling as shown below:
                makeErrorBoxes(xdata,ydata,xerrbox,yerrbox,fc='None',ec=nx_color,alpha=1.0,lw=0.5)
                makeErrorBoxes(xdata,ydata,xerrbox,yerrbox,fc=nx_color,ec='None',alpha=0.1,lw=0.5)
            '''
            xdata, ydata, xerror, yerror = np.array(xdata), np.array(ydata), np.array(xerror), np.array(yerror)
            # Create list for all the error patches
            errorboxes = []

            # Loop over data points; create box from errors at each point
            for xc, yc, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
                rect = Rectangle((xc - xe[0], yc - ye[0]), xe.sum(), ye.sum())
                errorboxes.append(rect)

            # Create patch collection with specified colour/alpha
            pc = PatchCollection(errorboxes, facecolor=fc, alpha=alpha, edgecolor=ec, linewidth=lw)

            # Add collection to axes
            ax.add_collection(pc)


        def r_squared(y, y_fit):
            '''
            Description:
                Calculate R^2 (R-squared) value between two sets of data, an experimental "y" and fitted "y_fit"

            Inputs:
                - `y` = list/array of y values (experimental)
                - `y_fit` = list/array of fitted y values to be compared against y

            Outputs:
                - `r_squared` = calculated R-squared value
            '''
            # Calculate R^2
            # residual sum of squares
            ss_res = np.sum((y - y_fit) ** 2)
            # total sum of squares
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            # r-squared
            r2 = 1 - (ss_res / ss_tot)
            return r2


        def chi_squared(y, y_fit, num_fit_params=0):
            '''
            Description:
                Calculate chi^2 (chi-squared) value between two sets of data, an experimental "y" and fitted "y_fit"

            Dependencies:
                `from scipy.stats import chisquare`

            Inputs:
                - `y` = list/array of y values (experimental)
                - `y_fit` = list/array of fitted y values to be compared against y
                - `num_deg_freedom` = number of degrees of freedom (DoF) in fit function (number of optimized parameters) (D=`0`)

            Outputs:
                - `chi_squared` = calculated chi-squared value
                - `reduced_chi_squared` = calculated reduced chi-squared value (chi^2 / DoF)
            '''
            # This normalization shouldn't be necessary, but a past build of scipy broke the chisquare function from working without it
            y_fit = y_fit * sum(y) / sum(y_fit)
            chi2, p = chisquare(y, f_exp=y_fit, ddof=num_fit_params)
            ndf = len(y) - num_fit_params
            # Hand calc
            # O_k = y     # observed, from measurement
            # E_k = y_fit # expected, from some distribution e.g. Gaussian
            # chi2 = np.sum( ((O_k-E_k)**2)/E_k )
            # num_deg_freedom = len(O_k) - num_fit_params
            # red_chi2 = chi2/num_deg_freedom
            return chi2, p, ndf

        def eval_distribution(x, dist_name='gauss', a=1, mu=0, sigma=0.3, a2=None, mu2=None, sigma2=None):
            '''
            Description:
                Evaluates a provided array of x values (or single value) for a desired distribution provided its defining parameters.

            Dependencies:
                `from munch import *`

            Inputs:
                - `x` = list/array of x values or single x value
                - `dist_name` = string denoting the distribution used (D=`'Gaussian'`), options include:
                               `['gauss','normal','Logistic','sech']`, Read more here on the [Gaussian/normal](https://en.wikipedia.org/wiki/Normal_distribution),
                               [Logistic](https://en.wikipedia.org/wiki/Logistic_distribution), and
                               [Hyperbolic secant](https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution) distributions.
                - `a` = distribution amplitude/height parameter
                - `mu` = distribution mean/center parameter
                - `sigma` = distribution width parameter

            Notes:
                At present, this function is only designed with "bell-shaped" distributions describable with three parameters related to
                amplitude/height, x position/centering, and width.

            Outputs:
                - `y_eval` = list/array of evaluated y values or single y value
                - `dist_info` = dictionary object containing the distribution `short_name` and `full_name`; assigned fit parameters `a`, `mu`, and `sigma`;
                              Python string of the function `fcn_py_str`; and LaTex string of the function `fcn_tex_str`
            '''
            dist_names_list = ['gauss', 'normal', 'Logistic', 'sech', 'cos', 'cosine', 'double-gauss']
            dist_name = dist_name.lower()
            if dist_name not in dist_names_list:
                print('Selected distribution name, ', dist_name, ' is not in the list of allowed distribution names: ',
                      dist_names_list, '\n exiting function... Please pick from this list and try again.')
                return None
            if type(x) is list: x = np.array(x)

            if dist_name == 'gauss' or dist_name == 'normal':
                y_eval = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
                fcn_py_str = 'f = a*np.exp(-(x-mu)**2/(2*sigma**2))'
                fcn_tex_str = r'f(x) = A$\cdot$exp$(\frac{-(x-\mu)^2}{2\sigma^2})$'
                if dist_name == 'gauss':
                    dist_full_name = 'Gaussian'
                else:
                    dist_full_name = 'Normal'
            elif dist_name == 'double-gauss':
                y_eval = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + a2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
                fcn_py_str = 'f = a_0*np.exp(-(x-mu_0)**2/(2*sigma_0**2)) + a_1*np.exp(-(x-mu_1)**2/(2*sigma_1**2))'
                fcn_tex_str = r'f(x) = $A_0\cdot$exp$(\frac{-(x-\mu_0)^2}{2\sigma_0^2})$' + '\n' + r'$\quad + A_1\cdot$exp$(\frac{-(x-\mu_1)^2}{2\sigma_1^2})$'
                dist_full_name = 'Double Gaussian'
            elif dist_name == 'logistic':
                y_eval = (a / (4 * sigma)) * (1 / np.cosh((x - mu) / (2 * sigma))) ** 2
                fcn_py_str = 'f = (a/(4*sigma))*(1/np.cosh((x-mu)/(2*sigma)))**2'
                fcn_tex_str = r'f(x) = $\frac{A}{4\sigma}$sech$^2(\frac{x-\mu}{2\sigma})$'
                dist_full_name = 'Logistic'
            elif dist_name == 'sech':
                y_eval = (a / (4 * sigma)) * (1 / np.cosh((x - mu) / (2 * sigma)))
                fcn_py_str = 'f = (a/(4*sigma))*(1/np.cosh((x-mu)/(2*sigma)))'
                fcn_tex_str = r'f(x) = $\frac{A}{4\sigma}$sech$(\frac{x-\mu}{2\sigma})$'
                dist_full_name = 'Hyperbolic secant'
            else:
                y_eval = a * np.cos(sigma * (x - mu))
                fcn_py_str = 'f = a*np.cos(sigma*(x-mu))'
                fcn_tex_str = r'f(x) = $A*\cos(\sigma*(x-\mu))$'
                dist_full_name = 'Cosine'

            dist_info = {'short_name': dist_name,
                         'full_name': dist_full_name,
                         'a': a,
                         'mu': mu,
                         'sigma': sigma,
                         'fcn_py_str': fcn_py_str,
                         'fcn_tex_str': fcn_tex_str}

            if dist_name == 'double-gauss':
                dist_info.update({
                    'a2': a2,
                    'mu2': mu2,
                    'sigma2': sigma2,
                })

            try:
                dist_info = Munch(dist_info)
            except:
                print("Munch failed.  Returned object is a conventional dictionary rather than a munch object.")

            return y_eval, dist_info


        def fit_distribution(x, y, dist_name='gauss', a0=None, mu0=None, sigma0=None, custom_fcn=None, a1=None, mu1=None,
                             sigma1=None):
            '''
            Description:
                Determine best fit parameters and quality of fit provided test x and y values, the name of the ditribution to be fit, and initial guesses of its parameters.
                If initial guesses are omitted, they will try to be automatically assessed (your mileage may vary).

            Dependencies:
                `from munch import *`
                `from scipy.optimize import curve_fit`
                `from lmfit.models import GaussianModel`

            Inputs:
                - `x` = list/array of x values to be fit
                - `y` = list/array of y values to be fit
                - `dist_name` = string denoting the distribution used (D=`'Gaussian'`), options include:
                               `['gauss','normal','Logistic','sech']`, Read more here on the [Gaussian/normal](https://en.wikipedia.org/wiki/Normal_distribution),
                               [Logistic](https://en.wikipedia.org/wiki/Logistic_distribution), and
                               [Hyperbolic secant](https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution) distributions.
                - `a0` = initial guess of the distribution amplitude/height parameter
                - `mu0` = initial guess of the distribution mean/center parameter
                - `sigma0` = initial guess of the distribution width parameter
                - `custom_fcn` = user-provided function which accepts arguments and can be called as `custom_fcn(x,a,mu,sigma)`, this will overwrite `dist_name`
                                 if set to anything other than its default value D=`None`

            Notes:
                At present, this function is only designed with "bell-shaped" distributions describable with three parameters related to
                amplitude/height, x position/centering, and width.

            Outputs:
                - `y_fit` = list/array of evaluated y values using the optimally found fit parameters
                - `dist_info` = dictionary object containing the distribution `short_name` and `full_name`; optimized fit parameters `a`, `mu`, and `sigma`;
                              calculated `FWHM`, `r2`/`r_squared` R^2, and `chi2`/`chi_squared` values; Python string of the function `fcn_py_str`; and LaTex string of the function `fcn_tex_str`
            '''
            dist_names_list = ['gauss', 'normal', 'Logistic', 'sech', 'cos', 'cosine', 'double-gauss']
            dist_name = dist_name.lower()
            if dist_name not in dist_names_list:
                print('Selected distribution name, ', dist_name, ' is not in the list of allowed distribution names: ',
                      dist_names_list, '\n exiting function... Please pick from this list and try again.')
                return None
            if type(x) is list: x = np.array(x)
            if type(y) is list: y = np.array(y)

            use_lmfit_for_gauss = False  # True

            def gaus(x, a, mu, sigma):
                f = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
                return f

            def double_gauss(x, a0, mu0, sigma0, a1, mu1, sigma1):
                f = a0 * np.exp(-(x - mu0) ** 2 / (2 * sigma0 ** 2)) + a1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
                return f

            def logistic_dist(x, a, mu, sigma):
                f = (a / (4 * sigma)) * (1 / np.cosh((x - mu) / (2 * sigma))) ** 2
                return f

            def hyperbolic_secant_dist(x, a, mu, sigma):
                f = (a / (4 * sigma)) * (1 / np.cosh((x - mu) / (2 * sigma)))
                return f

            def cos_dist(x, a, mu, sigma):
                f = a * np.cos(sigma * (x - mu))
                return f

            if custom_fcn != None:
                fit_fcn = custom_fcn
            else:
                if dist_name == 'gauss' or dist_name == 'normal':
                    fit_fcn = gaus
                elif dist_name == 'double-gauss':
                    fit_fcn = double_gauss
                elif dist_name == 'logistic':
                    fit_fcn = logistic_dist
                elif dist_name == 'cos' or dist_name == 'cosine':
                    fit_fcn = cos_dist
                else:  # if dist_name=='sech'
                    fit_fcn = hyperbolic_secant_dist

            n = len(x)
            ymax = max(y)
            ysum = sum(y)
            mean = sum(x * y / ymax) / (ysum / ymax)  # n
            sigma = sum((y / ymax) * (x - mean) ** 2) / (ysum / ymax)  # n

            if a0 == None: a0 = ymax
            if mu0 == None: mu0 = mean
            if sigma0 == None: sigma0 = sigma
            # print('GUESS MEAN, SIGMA:',mu0,sigma0)

            if use_lmfit_for_gauss and (dist_name == 'gauss' or dist_name == 'normal'):
                model = GaussianModel()
                # params = model.make_params(center=mu0, amplitude=a0, sigma=sigma0)
                params = model.guess(y, x=x)
                result = model.fit(y, params, x=x)
                # see: https://lmfit.github.io/lmfit-py/fitting.html
                a = result.params['amplitude'].value
                mu = result.params['center'].value
                sigma = result.params['sigma'].value
                FWHM_fit = 2 * np.sqrt(2 * np.log(2)) * sigma

                y_fit, dist_info = eval_distribution(x, dist_name=dist_name, a=a, mu=mu, sigma=sigma)
                r2 = r_squared(y, y_fit)

                chi2 = result.chisqr
                redchi2 = result.redchi
                num_fit_params = 3
                ndf_chi2 = len(y) - num_fit_params
                p_chi2 = stats.chi2.sf(chi2, ndf_chi2)


            else:
                if dist_name == 'double-gauss':
                    if a1 == None: a1 = a0
                    if mu1 == None: mu1 = mu0
                    if sigma1 == None: sigma1 = sigma0
                    popt, pcov = curve_fit(fit_fcn, x, y, p0=[a0, mu0, sigma0, a1, mu1, sigma1])
                    a, mu, sigma = popt[0], popt[1], abs(popt[2])
                    a2, mu2, sigma2 = popt[3], popt[4], abs(popt[5])
                    FWHM_fit = 2 * np.sqrt(2 * np.log(2)) * sigma

                    y_fit, dist_info = eval_distribution(x, dist_name=dist_name, a=a, mu=mu, sigma=sigma, a2=a2, mu2=mu2,
                                                         sigma2=sigma2)
                    r2 = r_squared(y, y_fit)
                else:
                    popt, pcov = curve_fit(fit_fcn, x, y, p0=[a0, mu0, sigma0])
                    a, mu, sigma = popt[0], popt[1], abs(popt[2])
                    FWHM_fit = 2 * np.sqrt(2 * np.log(2)) * sigma

                    y_fit, dist_info = eval_distribution(x, dist_name=dist_name, a=a, mu=mu, sigma=sigma)
                    r2 = r_squared(y, y_fit)

                # chi2,red_chi2 = chi_squared(y,y_fit,num_fit_params=3)
                num_fit_params = 3
                chi2, p_chi2, ndf_chi2 = chi_squared(y, y_fit, num_fit_params=num_fit_params)

            # How to write chi2 results: https://www.socscistatistics.com/tutorials/chisquare/default.aspx
            if p_chi2 < 0.001:
                chi_p_str = 'p<0.001'
            else:
                chi_p_str = 'p = {:4g}'
            chi2_tex_str = r'$\chi^2$' + '({:},N={:})={:.3g}, {}'.format(num_fit_params, n, chi2, chi_p_str)

            dist_info.update({
                'FWHM': FWHM_fit,
                'r2': r2,
                'r_squared': r2,
                'chi2': chi2,
                'chi_squared': chi2,
                'p_chi2': p_chi2,
                'ndf_chi2': ndf_chi2,
                'chi2_tex_str': chi2_tex_str
                # 'red_chi2':red_chi2,
                # 'red_chi_squared':red_chi2,
            })

            return y_fit, dist_info


        def rebinner(output_xbins, input_xbins, input_ybins):
            """
            Description:
                The purpose of this function is to rebin a set of y values corresponding to a set of x bins to a new set of x bins.
                The function seeks to be as generalized as possible, meaning bin sizes do not need to be consistent.

            Dependencies:
                `import numpy as np`

            Inputs:
              - `output_xbins` = output array containing bounds of x bins of length N; first entry is leftmost bin boundary
              - `input_xbins`  = input array containing bounds of x bins of length M; first entry is leftmost bin boundary
              - `input_ybins`  = input array containing y values of length M-1

            Outputs:
              - `output_ybins` = output array containing y values of length N-1
            """

            N = len(output_xbins)
            M = len(input_xbins)
            output_ybins = np.zeros(N - 1)

            for i in range(0, N - 1):
                # For each output bin
                lxo = output_xbins[i]  # lower x value of output bin
                uxo = output_xbins[i + 1]  # upper x value of output bin
                dxo = uxo - lxo  # width of current x output bin

                # Scan input x bins to see if any fit in this output bin
                for j in range(0, M - 1):
                    lxi = input_xbins[j]  # lower x value of input bin
                    uxi = input_xbins[j + 1]  # upper x value of input bin
                    dxi = uxi - lxi  # width of current x input bin

                    if uxi < lxo or lxi > uxo:
                        # no bins are aligned
                        continue
                    elif lxi >= lxo and lxi < uxo:
                        # start of an input bin occurs in this output bin
                        if lxi >= lxo and uxi <= uxo:
                            # input bin completely encompassed by output bin
                            output_ybins[i] = output_ybins[i] + input_ybins[j]
                        else:
                            # input bin spans over at least one output bin
                            # count fraction in current output x bin
                            f_in_dxo = (uxo - lxi) / dxi
                            output_ybins[i] = output_ybins[i] + f_in_dxo * input_ybins[j]
                    elif lxi < lxo and uxi > uxo:
                        # output bin is completely encompassed by input bin
                        f_in_dxo = (uxo - lxo) / dxi
                        output_ybins[i] = output_ybins[i] + f_in_dxo * input_ybins[j]
                    elif lxi < lxo and uxi > lxo and uxi <= uxo:
                        # tail of input bin is located in this output bin
                        f_in_dxo = (uxi - lxo) / dxi
                        output_ybins[i] = output_ybins[i] + f_in_dxo * input_ybins[j]

            return output_ybins


        def generate_line_bar_coordinates(xbins, yvals, yerrs=[]):
            """
            Description:
                Converts a set of bin boundaries and bin contents to coordinates mapping a bar plot if drawn with a line

            Inputs:
              - `xbins` = list of length N+1 bin boundary values
              - `yvals` = list of length N bin content values
              - `yerrs` = (optional) list of length N absolute uncertainties of bin content values

            Outputs:
              - `newx` = list of length 2N + 2 of x-coordinates mapping a 'bar plot' of the input histogram data
              - `newy` = list of length 2N + 2 of y-coordinates mapping a 'bar plot' of the input histogram data
              - `newyerr` = (optional) list of length 2N + 2 of y-coordinates mapping a 'bar plot' of the input histogram data
            """
            if len(yvals) != (len(xbins) - 1):
                pstr = 'xbins should be a list of bin edges of length one more than yvals, the values associated with the contents of each bin' + '\n'
                pstr += 'provided input arrays had lengths of {} for xbins and {} for yvals'.format(str(len(xbins)),
                                                                                                    str(len(yvals)))
                print(pstr)
                return 0
            newx = [xbins[0], xbins[0]]
            newy = [0, yvals[0]]
            if len(yerrs) != 0: newyerr = [0, yvals[0]]
            for i in range(len(xbins) - 2):
                newx.append(xbins[i + 1])
                newx.append(xbins[i + 1])
                newy.append(yvals[i])
                newy.append(yvals[i + 1])
                if len(yerrs) != 0:
                    newyerr.append(yerrs[i])
                    newyerr.append(yerrs[i + 1])
            newx.append(xbins[-1])
            newx.append(xbins[-1])
            newy.append(yvals[-1])
            newy.append(0)
            if len(yerrs) != 0:
                newyerr.append(yerrs[-1])
                newyerr.append(0)
                return newx, newy, newyerr
            else:
                return newx, newy

        def fancy_plot(
                # Required data
                xdata_lists, ydata_lists,

                # Dictionaries
                dictionaries=None,

                # Optional data
                data_labels=[], xerr_lists=[[]], yerr_lists=[[]],
                # Standard basic settings (optional)
                figi=1, title_str='title', x_label_str='x-axis', y_label_str='y-axis',
                x_limits=[], y_limits=[], x_scale='log', y_scale='log', color='#FDFEFC', alpha=1.0,
                linestyle='', linewidth=1,
                marker='.', markersize=5, markerfacecolor=None, markeredgecolor=None, markeredgewidth=None,
                errorstyle='bar-band', error_band_opacity=0.15,
                elinewidth=None, capsize=None,
                fig_width_inch=9.5, fig_height_inch=6.5, title_fs=16, axis_fs=14,
                f_family='sans-serif', f_style='normal', f_variant='normal', f_weight='normal',

                # Advanced settings (optional)
                # Legend settings
                legend_position='outside right', legend_anchor=None, legend_ncol=1, legend_alpha=None,
                legend_columnspacing=None,
                # Errorbar settings
                errorbox_xdata_l=[[]], errorbox_xdata_r=[[]], errorbox_ydata_l=[[]], errorbox_ydata_u=[[]],
                errorbox_fc='k', errorbox_fa=0.1, errorbox_ec='k', errorbox_ea=1.0, errorbox_ew=0.5,
                # Subplot settings
                fig=None, ax=None, spnrows=1, spncols=1, spindex=1,
                man_sp_placement=False, spx0=0.1, spy0=0.1, sph0=0.4, spw0=0.4
        ):
            '''
            Description:
                Function which makes very customizable and beautiful plots.  It is intended to be used when plotting multiple datasets at once with a legend but can also handle individual datasets

            Dependencies:
              - `import numpy as np`
              - `import matplotlib.pyplot as plt`

            Inputs:
              (Required)

              - `xdata_lists` = a list containing lists/arrays of x data (or single list of xdata applied to all ydata in y_data_lists)
              - `ydata_lists` = a list containing lists/arrays of y data (or single list of ydata)

              OR

              - `dictionaries` (see below)

            Dictionaries:
              - `dictionaries` = a list containing dictionary objects for each dataset to be plotted (or single dictionary object).

                    This provides an alternate way of providing this function with data to be plotted.  If wanting to use exclusively dictionaries,
                    set `xdata_lists=None` and `ydata_lists=None`; otherwise, the two modes may be used together.
                    The dictionaries are converted to the "standard" list of lists/strings/etc format native to this function.
                    Below are listed the input keywords for these dictionaries; where not the same as the normal variables for this function,
                    the equivalent name is provided in parentheses.

                    - Required: `'xdata'` (xdata_lists), `'ydata'` (ydata_lists)
                    - Optional (basic): `'data_label'` (data_labels), `'xerr'` (xerr_lists), `'yerr'` (yerr_lists),
                                `'color'`, `'alpha'`, `'linestyle'`, `'linewidth'`, `'marker'`, `'markersize'`,
                                `'markerfacecolor'`, `'markeredgecolor'`, `'markeredgewidth'`,
                                `'errorstyle'`, `'error_band_opacity'`, `'elinewidth'`, `'capsize'`
                    - Optional (advanced): `'errorbox_xdata_l'`, `'errorbox_xdata_r'`, `'errorbox_ydata_l'`, `'errorbox_ydata_u'`,
                                `'errorbox_fc'`, `'errorbox_fa'`, `'errorbox_ec'`, `'errorbox_ea'`, `'errorbox_ew'`

                    For any entry omitted from a dictionary, the value provided to the function is checked first; otherwise the default value is assumed.
                    For example, for entries missing the `'color'` keyword, the value provided to the color variable is used.  If it has not been changed from
                    its default value, then the default behavior is used.

            Inputs:
               (Optional, basic)

              - `data_labels` = a list of strings to be used as data labels in the legend (D=`[]`, no legend generated)
              - `xerr_lists` = a list containing lists/arrays of x data absolute uncertainties (or single list of xdata errors applied to all ydata in y_data_lists) (D=`[[]]`, No error)
              - `yerr_lists` = a list containing lists/arrays of y data absolute uncertainties (or single list of ydata errors) (D=`[[]]`, No error)
              - `figi` = figure index (D=`1`)
              - `title_str` = string to be used as the title of the plot (D=`'title'`)
              - `x_label_str` = string to be used as x-axis title (D=`'x-axis'`)
              - `y_label_str` = string to be used as y-axis title (D=`'y-axis'`)
              - `x_limits` = length 2 list specifying minimum and maximum x-axis bounds [xmin,xmax] (D=auto-calculated based on x_data_lists)
              - `y_limits` = length 2 list specifying minimum and maximum y-axis bounds [ymin,ymax] (D=auto-calculated based on y_data_lists)
              - `x_scale` = x-axis scale, either `"linear"`, `"log"`, `"symlog"`, or `"logit"`
              - `y_scale` = y-axis scale, either `"linear"`, `"log"`, `"symlog"`, or `"logit"`
              - `color` = list of color strings to be used of same length as y_data_lists (or individual color string) (D=Matplotlib default color cycle)
              - `alpha` = list of (or individual) alpha values (D=`1.0`)
              - `linestyle` = list of (or individual) strings denoting linestyle: `''`, `'-'`, `'--'`, `'-.'`, or `':'` (D=`''`)
              - `linewidth` = list of (or individual) int/float of the width of line (D=`1`)
              - `marker` = list of (or individual) marker styles (D=`'.'`) For all options, see: https://matplotlib.org/3.1.0/api/markers_api.html
              - `markersize` = list of (or individual) int/float of marker size (D=`5`)
              - `markerfacecolor` = list of (or individual) marker face colors (D=`None`, use value of `'color'`)
              - `markeredgecolor` = list of (or individual) marker edge colors (D=`None`, use value of `'color'`)
              - `markeredgewidth` = list of (or individual) int/float of marker edge widths (D=`None`)
              - `errorstyle` = list of (or individual) strings specifying how y-error is represented (D=`'bar-band'`, `['bar-band','bar','band']`)
              - `error_band_opacity` = list of (or individual) int/float of error band opacities (D=`0.15`)
              - `elinewidth` = list of (or individual) int/float  line width of error bar lines (D=`None`, use current `linewidth`)
              - `capsize` = list of (or individual) int/float of length of the error bar caps in points (D=`None`)
              - `fig_width_inch` = figure width in inches (D=`9.5`)
              - `fig_height_inch` = figure height in inches (D=`6.5`)
              - `title_fs` = title font size (D=`16`)
              - `axis_fs` = axis label font size (D=`14`)
              - `f_family` = string specifying font family (D=`'sans-serif'`); options include: `['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']`
              - `f_style` = string specifying font style (D=`'normal'`); options include: `['normal', 'italic', 'oblique']`
              - `f_variant` = string specifying font variant (D=`'normal'`); options include: `['normal', 'small-caps']`
              - `f_weight` = string specifying font weight (D=`'normal'`); options include: `['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']`


            Inputs:
               (Optional, advanced)

               Legend settings

              - `legend_position` = one of the default matplotlib legend position strings (`'best'`,`'upper right'`,`'lower center'`,`'lower left'`,etc.) to place the legend inside the plot or
                                `'outside right'` or `'outside bottom'` to place the legend outside of the plot area (D=`'outside right'`, if legend is to be used)
              - `legend_anchor` = legend anchor position (x=left-right position, y=bottom-top position) only used when legend position is set to one of the "outside" options
                                (D=`None` which becomes `(1.0,0.75)` if position is `'outside right'` or `(0.5,-0.17)` if position is `'outside bottom'`)
                                Note that only one coordinate usually should be adjusted.  If using an `'outside right'` legend, only the y-coordinate needs to be manipulated
                                to scoot the legend up/down.  Likewise, for `'outside bottom'` legends only the x-coordinate needs adjusting for tuning left/right position
              - `legend_ncol` = number of columns in legend (D=`1` in all cases except for legend_position=`'outside bottom'` where D=`len(ydata_lists)`)
              - `legend_alpha` = alpha of legend background (D=`None`, auto determined by matplotlib)
              - `legend_columnspacing` = column spacing of legend (D=`None`, auto determined by matplotlib)

               Error boxes (can be used in addition to or in lieu of normal error bars)

              - `errorbox_xdata_l` = a list containing lists/arrays of errorbox left widths from center (x-data lower error)
              - `errorbox_xdata_r` = a list containing lists/arrays of errorbox right widths from center (x-data upper error)
              - `errorbox_ydata_l` = a list containing lists/arrays of errorbox lower heights from center (y-data lower error)
              - `errorbox_ydata_u` = a list containing lists/arrays of errorbox upper heights from center (y-data upper error)

                  *Error boxes will only be drawn if at least one x list and one y list of the four above arrays is specified; unspecified lists will default to zero error.*

              - `errorbox_fc` = list of (or individual) error box face color (D=`'k'`, black)
              - `errorbox_fa` = list of (or individual) error box face alpha (D=`0.1`)
              - `errorbox_ec` = list of (or individual) error box edge color (D=`'k'`, black)
              - `errorbox_ea` = list of (or individual) error box edge alpha (D=`1.0`)
              - `errorbox_ew` = list of (or individual) error box edge width (D=`0.5`)

               Subplots

              - `fig` = figure handles from existing figure to draw on (D=`None`, `fig=None` should always be used for initial subplot unless a figure canvas has already been generated)
              - `ax` = axis handles from an existing figure to draw on (D=`None`, `ax=None` should always be used for initial subplot)
              - `spnrows` = number of rows in final subplot (D=`1`)
              - `spncols` = number of columns in final subplot (D=`1`)
              - `spindex` = index of current subplot (between 1 and spnrows*spncols) (D=`1`)
              - `man_sp_placement` = logical variable controlling manual sizing/placement of subplots using below variables (D=`False`, use automatic sizing)
              - `spx0` = distance from canvas left edge where this plotting area should begin (D=`0.1`), generally a number around 0~1
              - `spy0` = distance from canvas bottom edge where this plotting area should begin (D=`0.1`), generally a number around 0~1
              - `spw0` = width of this plotting area on the canvas (D=`0.4`), generally a number around 0~1
              - `sph0` = height of this plotting area on the canvas (D=`0.4`), generally a number around 0~1

            Outputs:
              - `fig` = pyplot figure
              - `ax`  = pyplot figure plot/subplot axes handles
            '''
            include_legend = True  # used to toggle legend on/off
            single_dataset = False  # Assume multiple datasets entered, but this can be tested to see if it is the case or not.

            # At the very start, check for dictionaries
            if dictionaries != None:
                # determine if single entry or list of entries
                if isinstance(dictionaries, dict):
                    dictionaries = [dictionaries]
                num_dict = len(dictionaries)

                dxdata_lists = []
                dydata_lists = []
                ddata_labels = []
                dxerr_lists = []
                dyerr_lists = []

                dcolor, dalpha, dlinestyle, dlinewidth, dmarker, dmarkersize, dmarkerfacecolor, dmarkeredgecolor, dmarkeredgewidth = [], [], [], [], [], [], [], [], []
                derrorstyle, derror_band_opacity, delinewidth, dcapsize = [], [], [], []
                derrorbox_xdata_l, derrorbox_xdata_r, derrorbox_ydata_l, derrorbox_ydata_u = [], [], [], []
                derrorbox_fc, derrorbox_fa, derrorbox_ec, derrorbox_ea, derrorbox_ew = [], [], [], [], []

                keylist = ['xdata', 'xdata_lists', 'ydata', 'ydata_lists', 'data_label', 'data_labels', 'xerr',
                           'xerr_lists', 'yerr', 'yerr_lists',
                           'color', 'alpha', 'linestyle', 'linewidth', 'marker', 'markersize', 'markerfacecolor',
                           'markeredgecolor', 'markeredgewidth',
                           'errorstyle', 'error_band_opacity', 'elinewidth, capsize',
                           'errorbox_xdata_l', 'errorbox_xdata_r', 'errorbox_ydata_l', 'errorbox_ydata_u',
                           'errorbox_fc', 'errorbox_fa', 'errorbox_ec', 'errorbox_ea', 'errorbox_ew']

                settings_realvars = [color, alpha, linestyle, linewidth, marker, markersize, markerfacecolor,
                                     markeredgecolor, markeredgewidth, errorstyle, error_band_opacity, elinewidth, capsize,
                                     errorbox_xdata_l, errorbox_xdata_r, errorbox_ydata_l, errorbox_ydata_u, errorbox_fc,
                                     errorbox_fa, errorbox_ec, errorbox_ea, errorbox_ew]
                settings_keys = ['color', 'alpha', 'linestyle', 'linewidth', 'marker', 'markersize', 'markerfacecolor',
                                 'markeredgecolor', 'markeredgewidth', 'errorstyle', 'error_band_opacity', 'elinewidth',
                                 'capsize', 'errorbox_xdata_l', 'errorbox_xdata_r', 'errorbox_ydata_l', 'errorbox_ydata_u',
                                 'errorbox_fc', 'errorbox_fa', 'errorbox_ec', 'errorbox_ea', 'errorbox_ew']
                settings_vars = [dcolor, dalpha, dlinestyle, dlinewidth, dmarker, dmarkersize, dmarkerfacecolor,
                                 dmarkeredgecolor, dmarkeredgewidth, derrorstyle, derror_band_opacity, delinewidth,
                                 dcapsize, derrorbox_xdata_l, derrorbox_xdata_r, derrorbox_ydata_l, derrorbox_ydata_u,
                                 derrorbox_fc, derrorbox_fa, derrorbox_ec, derrorbox_ea, derrorbox_ew]
                settings_defalts = ['#FDFEFC', 1.0, '', 1, '.', 5, None, None, None, 'bar-band', 0.15, None, None, [], [],
                                    [], [], 'k', 0.1, 'k', 1.0, 0.5]

                for i in range(num_dict):
                    d = dictionaries[i]
                    if not isinstance(d, dict):
                        print('Index {} of dictionaries list is not a dictionary! Quitting...'.format(i))
                        return None

                    # Check for any unrecognizable keys
                    dkeys = list(d.keys())
                    for dkey in dkeys:
                        if dkey not in keylist:
                            print('Encountered unknown keyword {} in dictionary entry at index {}.  Ignoring it...'.format(
                                dkey, i))

                    # Check for each key that will be used
                    if 'xdata' in d:
                        dxdata_lists.append(d['xdata'])
                    elif 'xdata_lists' in d:
                        dxdata_lists.append(d['xdata_lists'])
                    else:
                        print('Dictionary at index {} is missing xdata.  Quitting...'.format(i))
                        return None

                    if 'ydata' in d:
                        dydata_lists.append(d['ydata'])
                    elif 'ydata_lists' in d:
                        dydata_lists.append(d['ydata_lists'])
                    else:
                        print('Dictionary at index {} is missing ydata.  Quitting...'.format(i))
                        return None

                    if 'data_label' in d:
                        ddata_labels.append(d['data_label'])
                    elif 'data_labels' in d:
                        ddata_labels.append(d['data_labels'])
                    else:
                        ddata_labels.append(None)

                    if 'xerr' in d:
                        dxerr_lists.append(d['xerr'])
                    elif 'xerr_lists' in d:
                        dxerr_lists.append(d['xerr_lists'])
                    else:
                        dxerr_lists.append([])

                    if 'yerr' in d:
                        dyerr_lists.append(d['yerr'])
                    elif 'yerr_lists' in d:
                        dyerr_lists.append(d['yerr_lists'])
                    else:
                        dyerr_lists.append([])

                    for ski in range(len(settings_keys)):
                        if settings_keys[ski] in d:
                            settings_vars[ski].append(d[settings_keys[ski]])
                        elif not isinstance(settings_realvars[ski], (
                        list, np.ndarray)):  # if main entry is not a list, use it instead of default value
                            settings_vars[ski].append(settings_realvars[ski])
                        else:
                            settings_vars[ski].append(settings_defalts[ski])

                # Now combine with data entered normally, if applicable
                if xdata_lists != None and ydata_lists != None:
                    # combine
                    if not isinstance(xdata_lists[0], (list, np.ndarray)):
                        xdata_lists = [xdata_lists]
                    xdata_lists = xdata_lists + dxdata_lists

                    if not isinstance(ydata_lists[0], (list, np.ndarray)):
                        ydata_lists = [ydata_lists]
                    num_normal_datasets = len(ydata_lists)
                    num_dict_datasets = len(dydata_lists)
                    ydata_lists = ydata_lists + dydata_lists

                    if data_labels == []:
                        if not all(x == None for x in ddata_labels):
                            data_labels = num_normal_datasets * [None] + ddata_labels
                    else:
                        data_labels = data_labels + ddata_labels

                    if xerr_lists == [[]]:
                        if not all(x == [] for x in dxerr_lists):
                            xerr_lists = num_normal_datasets * [[]] + dxerr_lists
                    else:
                        xerr_lists = xerr_lists + dxerr_lists

                    if yerr_lists == [[]]:
                        if not all(x == [] for x in dyerr_lists):
                            yerr_lists = num_normal_datasets * [[]] + dyerr_lists
                    else:
                        yerr_lists = yerr_lists + dyerr_lists

                    for ski in range(len(settings_keys)):
                        if settings_keys[ski] in ['errorbox_xdata_l', 'errorbox_xdata_r', 'errorbox_ydata_l',
                                                  'errorbox_ydata_u']:
                            # the special exceptions which can be lists
                            if settings_realvars[ski] == [[]]:
                                if not all(x == [] for x in settings_vars[ski]):
                                    settings_realvars[ski] = num_normal_datasets * [[]] + settings_vars[ski]
                            else:
                                settings_realvars[ski] = settings_realvars[ski] + settings_vars[ski]
                        else:
                            # for each possible setting option which could be a single value or list
                            if not isinstance(settings_realvars[ski], (list, np.ndarray)):  # if main entry isn't a list
                                if not all(x == settings_realvars[ski] for x in
                                           settings_vars[ski]):  # if not all dict entries are same as main entry
                                    settings_realvars[ski] = num_normal_datasets * [settings_realvars[ski]] + settings_vars[
                                        ski]
                            else:  # just combine the two lists
                                # print(settings_vars[ski])
                                settings_realvars[ski] = settings_realvars[ski] + settings_vars[ski]
                    color, alpha, linestyle, linewidth, marker, markersize, markerfacecolor, markeredgecolor, markeredgewidth, errorstyle, error_band_opacity, elinewidth, capsize, errorbox_xdata_l, errorbox_xdata_r, errorbox_ydata_l, errorbox_ydata_u, errorbox_fc, errorbox_fa, errorbox_ec, errorbox_ea, errorbox_ew = settings_realvars

                else:  # the only data present are in dictionary form
                    xdata_lists = dxdata_lists
                    ydata_lists = dydata_lists

                    if all([x == None for x in data_labels]):
                        data_labels = None
                    else:
                        data_labels = ddata_labels

                    xerr_lists = dxerr_lists
                    yerr_lists = dyerr_lists

                    for ski in range(len(settings_keys)):
                        if settings_keys[ski] in ['errorbox_xdata_l', 'errorbox_xdata_r', 'errorbox_ydata_l',
                                                  'errorbox_ydata_u']:
                            # the special exceptions which can be lists
                            if all(x == [] for x in settings_vars[ski]):
                                settings_vars[ski] = [[]]  # set the error box parameters to appear as expected if empty
                    color, alpha, linestyle, linewidth, marker, markersize, markerfacecolor, markeredgecolor, markeredgewidth, errorstyle, error_band_opacity, elinewidth, capsize, errorbox_xdata_l, errorbox_xdata_r, errorbox_ydata_l, errorbox_ydata_u, errorbox_fc, errorbox_fa, errorbox_ec, errorbox_ea, errorbox_ew = settings_vars

            # End of dictionary entry handling

            # Determine if error boxes are to be drawn
            draw_error_boxes = False
            if (errorbox_xdata_l != [[]] or errorbox_xdata_r != [[]]) and (
                    errorbox_ydata_l != [[]] or errorbox_ydata_u != [[]]):
                draw_error_boxes = True

            if (not xdata_lists) and (not ydata_lists):
                print('Warning: Both xdata and ydata lists are empty (figure index = {}, titled "{}")'.format(figi,
                                                                                                              title_str))
                single_dataset = True
                include_legend = False
                xdata_lists = [[]]
                ydata_lists = []
                xerr_lists = [[]]
                yerr_lists = []
            elif (not xdata_lists):
                print('Warning: xdata list is empty (figure index = {}, titled "{}")'.format(figi, title_str))
            elif (not ydata_lists):
                print('Warning: ydata list is empty (figure index = {}, titled "{}")'.format(figi, title_str))

            # If using a single dataset (user inputs a single list, not a list of list(s)
            # if len(np.shape(ydata_lists)) != 1: # not just a simple list
            if (all(isinstance(el, (int, float)) for el in
                    ydata_lists)):  # input ydata is a single dataset, not a list of lists/arrays, convert to a list containing a single list for compatability with remainder of code
                single_dataset = True
                ydata_lists = [ydata_lists]
                yerr_lists = [yerr_lists]
                include_legend = False
                if draw_error_boxes:
                    errorbox_xdata_l = [errorbox_xdata_l]
                    errorbox_xdata_r = [errorbox_xdata_r]
                    errorbox_ydata_l = [errorbox_ydata_l]
                    errorbox_ydata_u = [errorbox_ydata_u]

            if not data_labels: include_legend = False

            nds = len(ydata_lists)

            # Allow use of single set of xdata for multiple sets of ydata
            if (not single_dataset) and (all(isinstance(el, (int, float)) for el in
                                             xdata_lists)):  # ydata is list of lists, xdata is a single list.  Assume same xdata for each set of ydata
                xdata2 = []
                for i in range(nds):
                    xdata2.append(xdata_lists)
                xdata_lists = xdata2

                if (all(isinstance(el, (int, float)) for el in
                        xerr_lists)):  # ydata is list of lists, xerr_data is a single list.  Assume same xerr_data for each set of ydata
                    xerr2 = []
                    for i in range(nds):
                        xerr2.append(xerr_lists)
                    xerr_lists = xerr2

                if draw_error_boxes:
                    errorbox_xdata_l2 = []
                    errorbox_xdata_r2 = []
                    for i in range(nds):
                        errorbox_xdata_l2.append(errorbox_xdata_l)
                        errorbox_xdata_r2.append(errorbox_xdata_r)
                    errorbox_xdata_l = errorbox_xdata_l2
                    errorbox_xdata_r = errorbox_xdata_r2

            fst = title_fs  # 16
            fs = axis_fs  # 14
            y_min = 1.0e10  # later used to set y-axis minimum
            y_max = 1.0e-14  # later used to set y-axis maximum
            x_min = 1.0e5  # later used to set x-axis minimum
            x_max = 1.0e1  # later used to set x-axis maximum

            plt.rc('font', family=f_family, style=f_style, variant=f_variant, weight=f_weight)

            if fig == None:
                fig = plt.figure(figi)
            # bg_color = '#FFFFFF' #'#E1E4E6'
            # fig.patch.set_facecolor(bg_color)
            # fig.patch.set_alpha(1.0)
            ax = plt.subplot(int(spnrows), int(spncols), int(spindex))

            for i in range(nds):
                xdata = xdata_lists[i]
                ydata = np.array(ydata_lists[i])
                xerr = None
                yerr = None
                xerr_present = False
                yerr_present = False
                if len(xerr_lists[0]) > 0:
                    xerr = xerr_lists[i]
                    if np.sum(xerr) == 0:
                        xerr = None
                    else:
                        xerr_present = True
                if len(yerr_lists[0]) > 0:
                    yerr = yerr_lists[i]
                    if np.sum(yerr) == 0:
                        yerr = None
                    else:
                        yerr_present = True

                if include_legend:
                    label_str = data_labels[i]
                else:
                    label_str = ''

                # Get settings which may be constant or vary by dataset (lists)
                if isinstance(color, (list, np.ndarray)):
                    c = color[i]
                else:
                    c = color
                if isinstance(alpha, (list, np.ndarray)):
                    alp = alpha[i]
                else:
                    alp = alpha
                if isinstance(linestyle, (list, np.ndarray)):
                    ls = linestyle[i]
                else:
                    ls = linestyle
                if isinstance(linewidth, (list, np.ndarray)):
                    lw = linewidth[i]
                else:
                    lw = linewidth
                if isinstance(marker, (list, np.ndarray)):
                    mkr = marker[i]
                else:
                    mkr = marker
                if isinstance(markersize, (list, np.ndarray)):
                    mks = markersize[i]
                else:
                    mks = markersize
                if isinstance(errorstyle, (list, np.ndarray)):
                    ers = errorstyle[i]
                else:
                    ers = errorstyle
                if isinstance(error_band_opacity, (list, np.ndarray)):
                    ebo = error_band_opacity[i]
                else:
                    ebo = error_band_opacity
                if isinstance(elinewidth, (list, np.ndarray)):
                    elw = elinewidth[i]
                else:
                    elw = elinewidth
                if isinstance(capsize, (list, np.ndarray)):
                    ecs = capsize[i]
                else:
                    ecs = capsize
                if isinstance(markerfacecolor, (list, np.ndarray)):
                    mfc = markerfacecolor[i]
                else:
                    mfc = markerfacecolor
                if isinstance(markeredgecolor, (list, np.ndarray)):
                    mec = markeredgecolor[i]
                else:
                    mec = markeredgecolor
                if isinstance(markeredgewidth, (list, np.ndarray)):
                    mew = markeredgewidth[i]
                else:
                    mew = markeredgewidth

                # Make actual plot
                if (not xerr_present and not yerr_present) or (ers == 'band' and not xerr_present):
                    if color == '#FDFEFC' or color[
                        0] == '#FDFEFC':  # assume user will never actually want/input this specific white color
                        p = ax.plot(xdata, ydata, label=label_str, ls=ls, lw=lw, marker=mkr, ms=mks, mfc=mfc, mec=mec,
                                    mew=mew, alpha=alp)
                    else:
                        p = ax.plot(xdata, ydata, label=label_str, c=c, ls=ls, lw=lw, marker=mkr, ms=mks, mfc=mfc, mec=mec,
                                    mew=mew, alpha=alp)
                else:
                    if color == '#FDFEFC' or color[
                        0] == '#FDFEFC':  # assume user will never actually want/input this specific white color
                        p = ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, label=label_str, ls=ls, lw=lw, marker=mkr,
                                        ms=mks, elinewidth=elw, capsize=ecs, mfc=mfc, mec=mec, mew=mew, alpha=alp)
                    else:
                        p = ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, label=label_str, c=c, ls=ls, lw=lw, marker=mkr,
                                        ms=mks, elinewidth=elw, capsize=ecs, mfc=mfc, mec=mec, mew=mew, alpha=alp)

                if (ers == 'bar-band' or ers == 'band') and (yerr_present or xerr_present):
                    if color == '#FDFEFC' or color[
                        0] == '#FDFEFC':  # assume user will never actually want/input this specific white color
                        c = p[0].get_color()  # need to grab whatever color was just used
                    if yerr_present:
                        if len(np.shape(yerr)) == 1:
                            ax.fill_between(xdata, np.array(ydata) - np.array(yerr), np.array(ydata) + np.array(yerr),
                                            color=c, alpha=ebo)
                        else:
                            ax.fill_between(xdata, np.array(ydata) - np.array(yerr[0, :]),
                                            np.array(ydata) + np.array(yerr[1, :]), color=c, alpha=ebo)
                    else:  # Fix this later to also accomodate cases where both x and y error are present?
                        if len(np.shape(xerr)) == 1:
                            ax.fill_betweenx(ydata, np.array(xdata) - np.array(xerr), np.array(xdata) + np.array(xerr),
                                             color=c, alpha=ebo)
                        else:
                            ax.fill_betweenx(ydata, np.array(xdata) - np.array(xerr[0, :]),
                                             np.array(xdata) + np.array(xerr[1, :]), color=c, alpha=ebo)

                if draw_error_boxes:
                    draw_error_box_for_this_dataset = True
                    # Ensure x and y error arrays are correctly sized, accounting for possible datasets without error boxes
                    # determine which, if either, x errors are 'None'
                    if (errorbox_xdata_l != [[]] and errorbox_xdata_r == [[]]):
                        erb_x_l = errorbox_xdata_l[i]
                        if len(erb_x_l) != len(ydata):
                            draw_error_box_for_this_dataset = False
                        else:
                            erb_x_r = 0 * np.array(erb_x_l)
                    elif (errorbox_xdata_l == [[]] and errorbox_xdata_r != [[]]):
                        erb_x_r = errorbox_xdata_r[i]
                        if len(erb_x_r) != len(ydata):
                            draw_error_box_for_this_dataset = False
                        else:
                            erb_x_l = 0 * np.array(erb_x_r)
                    else:  # both datasets possibly present
                        erb_x_l = errorbox_xdata_l[i]
                        erb_x_r = errorbox_xdata_r[i]
                        if len(erb_x_l) != len(ydata) and len(erb_x_r) != len(ydata):
                            draw_error_box_for_this_dataset = False
                        elif len(erb_x_l) != len(ydata) or len(erb_x_r) != len(ydata):
                            if len(erb_x_l) != len(ydata):
                                erb_x_l = 0 * np.array(erb_x_r)
                            elif len(erb_x_r) != len(ydata):
                                erb_x_r = 0 * np.array(erb_x_l)

                    # determine which, if either, y errors are 'None'
                    if (errorbox_ydata_l != [[]] and errorbox_ydata_u == [[]]):
                        erb_y_l = errorbox_ydata_l[i]
                        if len(erb_y_l) != len(ydata):
                            draw_error_box_for_this_dataset = False
                        else:
                            erb_y_u = 0 * np.array(erb_y_l)
                    elif (errorbox_ydata_l == [[]] and errorbox_ydata_u != [[]]):
                        erb_y_u = errorbox_ydata_u[i]
                        if len(erb_y_u) != len(ydata):
                            draw_error_box_for_this_dataset = False
                        else:
                            erb_y_l = 0 * np.array(erb_y_u)
                    else:  # both datasets possibly present
                        erb_y_l = errorbox_ydata_l[i]
                        erb_y_u = errorbox_ydata_u[i]
                        if len(erb_y_l) != len(ydata) and len(erb_y_u) != len(ydata):
                            draw_error_box_for_this_dataset = False
                        elif len(erb_y_l) != len(ydata) or len(erb_y_u) != len(ydata):
                            if len(erb_y_l) != len(ydata):
                                erb_y_l = 0 * np.array(erb_y_u)
                            elif len(erb_y_u) != len(ydata):
                                erb_y_u = 0 * np.array(erb_y_l)

                    if draw_error_box_for_this_dataset:
                        xerrbox = [erb_x_l, erb_x_r]
                        yerrbox = [erb_y_l, erb_y_u]

                        # Get settings which may be constant or vary by dataset (lists)
                        if isinstance(errorbox_fc, (list, np.ndarray)):
                            efc = errorbox_fc[i]
                        else:
                            efc = errorbox_fc
                        if isinstance(errorbox_ec, (list, np.ndarray)):
                            eec = errorbox_ec[i]
                        else:
                            eec = errorbox_ec
                        if isinstance(errorbox_ea, (list, np.ndarray)):
                            eea = errorbox_ea[i]
                        else:
                            eea = errorbox_ea
                        if isinstance(errorbox_fa, (list, np.ndarray)):
                            efa = errorbox_fa[i]
                        else:
                            efa = errorbox_fa
                        if isinstance(errorbox_ew, (list, np.ndarray)):
                            eew = errorbox_ew[i]
                        else:
                            eew = errorbox_ew

                        makeErrorBoxes(ax, xdata, ydata, xerrbox, yerrbox, fc='None', ec=eec, alpha=eea, lw=eew)  # outline
                        makeErrorBoxes(ax, xdata, ydata, xerrbox, yerrbox, fc=efc, ec='None', alpha=efa, lw=eew)  # fill face

                if len(ydata) != 0:
                    if all([yi == None for yi in ydata]):
                        print("\t\tfancy_plot warning: Encountered set of only 'None' at index {}".format(i))
                    elif len(ydata[np.nonzero(ydata)]) != 0:
                        if min(ydata[np.nonzero(ydata)]) < y_min: y_min = min(ydata[np.nonzero(ydata)])
                        # if min(ydata)<y_min: y_min = min(ydata)
                        if max(ydata[ydata != None]) > y_max: y_max = max(ydata[ydata != None])
                        if min(xdata) < x_min: x_min = min(xdata)
                        if max(xdata) > x_max: x_max = max(xdata)

            if title_str.strip() != '':
                window_title = slugify(title_str)  # "comparison_fig"
            else:
                window_title = 'Figure ' + str(figi)
            # window_title = window_title.replace('b','',1) # remove leading 'b' character from slugify process
            fig.canvas.manager.set_window_title(window_title)

            # hangle figure/legend positioning/sizing
            # First, figure size
            default_fig_x_in = fig_width_inch
            default_fig_y_in = fig_height_inch
            fig_x_in = default_fig_x_in
            fig_y_in = default_fig_y_in
            fig.set_size_inches(fig_x_in, fig_y_in)

            mpl_leg_pos_names = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                                 'center right', 'lower center', 'upper center', 'center']
            custom_leg_pos_names = ['outside right', 'outside bottom']

            if include_legend and legend_position in custom_leg_pos_names:
                if legend_anchor == None:
                    if legend_position == 'outside right':
                        legend_anchor = (1.0, 0.75)
                    elif legend_position == 'outside bottom':
                        legend_anchor = (0.5, -0.17)
                leg1_anchor = legend_anchor  # varied items
                handles_l1, labels_l1 = ax.get_legend_handles_labels()
                if legend_position == 'outside right':
                    legend1 = ax.legend(handles_l1, labels_l1, loc='upper left', bbox_to_anchor=leg1_anchor,
                                        ncol=legend_ncol, framealpha=legend_alpha, columnspacing=legend_columnspacing)
                elif legend_position == 'outside bottom':
                    if legend_ncol == 1 and len(data_labels) > 1: legend_ncol = len(data_labels)
                    legend1 = ax.legend(handles_l1, labels_l1, loc='upper center', bbox_to_anchor=leg1_anchor,
                                        ncol=legend_ncol, framealpha=legend_alpha, columnspacing=legend_columnspacing)
                ax.add_artist(legend1)
                fig.canvas.draw()
                f1 = legend1.get_frame()
                l1_w0_px, l1_h0_px = f1.get_width(), f1.get_height()
                l_w0_in, l_h0_in = l1_w0_px / fig.dpi, l1_h0_px / fig.dpi  # width and height of legend, in inches
            else:
                l_w0_in, l_h0_in = 0.0, 0.0
                if include_legend and legend_position not in custom_leg_pos_names:  # use matplotlib default-style legend inside plot area
                    ax.legend(loc=legend_position, ncol=legend_ncol, framealpha=legend_alpha,
                              columnspacing=legend_columnspacing)

            n_title_lines = 0
            if title_str.strip() != '':
                n_title_lines = 1 + title_str.count('\n')
            n_xlabel_lines = 0
            if x_label_str.strip() != '':
                n_xlabel_lines = 1 + x_label_str.count('\n')
            n_ylabel_lines = 0
            if y_label_str.strip() != '':
                n_ylabel_lines = 1 + y_label_str.count('\n')

            # These values are good, do not change them.  (derived while working on SHAEDIT project)
            x0bar = 0.60 + 0.200 * n_ylabel_lines  # inches, horizontal space needed for ylabel
            y0bar = 0.45 + 0.200 * n_xlabel_lines  # inches, vertical space needed for xticks/numbers, xlabel and any extra lines it has
            t0bar = 0.10 + 0.300 * n_title_lines  # inches, vertical space needed for title
            del_l_in = 0.15  # inches, extra horizontal padding right of legend

            # adjust legend spacing depending on its position
            if legend_position == 'outside right':
                l_h0_in = 0.0
            elif legend_position == 'outside bottom':
                l_w0_in = 0.0

            # Plot window placement and sizing
            x0 = x0bar / fig_x_in  # distance from left edge that plot area begins
            y0 = y0bar / fig_y_in + (l_h0_in / fig_y_in)  # distance from bottom edge that plot area begins
            h0 = 1 - (y0bar + t0bar) / fig_y_in - (
                        l_h0_in / fig_y_in)  # height of plot area, set to be full height minus space needed for title, x-label, and potentially an outside bottom legend
            w0 = 1 - x0 - (l_w0_in / fig_x_in) - (
                        del_l_in / fig_x_in)  # width of plot area, set to be full width minus space needed for y-label and potentially an outside right legend

            if man_sp_placement:
                if spx0 != None: x0 = spx0
                if spy0 != None: y0 = spy0
                if sph0 != None: h0 = sph0
                if spw0 != None: w0 = spw0

            # Set size and location of the plot on the canvas
            box = ax.get_position()
            # all vals in [0,1]: left, bottom, width, height
            if not man_sp_placement and (spnrows != 1 or spncols != 1):
                pstr = 'Warning: It is highly encouraged that subplots be positioned manually.\n'
                pstr += '   This is done by setting man_sp_placement=True and then adjusting\n'
                pstr += '   the parameters spx0, spy0, sph0, and spw0 for each subplot.\n'
                pstr += '   The current plot was automatically sized by matplotlib.\n'
                print(pstr)
            else:
                ax.set_position([x0, y0, w0, h0])

            plt.title(title_str, fontsize=fst)
            plt.xlabel(x_label_str, fontsize=fs)
            plt.ylabel(y_label_str, fontsize=fs)
            plt.xscale(x_scale)
            plt.yscale(y_scale)
            plt.grid(visible=True, which='major', linestyle='-', alpha=0.25)
            plt.grid(visible=True, which='minor', linestyle='-', alpha=0.10)
            # ensure at least minimum number of decades are present on a plot by increasing padding if necessary
            zoom_mult = 1.0
            x_log_buffer = 0.15 * zoom_mult
            y_log_buffer = 0.2 * zoom_mult
            min_x_decs = 2
            min_y_decs = 2

            x_scale = 'linear'

            if not x_limits:
                if x_scale == 'log':  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                    if (np.log10(x_max) - np.log10(x_min) + 2 * x_log_buffer) < min_x_decs:
                        x_log_buffer = 0.5 * (min_x_decs - (np.log10(x_max) - np.log10(x_min)))
                    plt.xlim([10 ** (np.log10(x_min) - x_log_buffer), 10 ** (np.log10(x_max) + x_log_buffer)])
            else:
                plt.xlim(x_limits)

            if not y_limits:
                if y_scale == 'log':  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                    if (np.log10(y_max) - np.log10(y_min) + 2 * y_log_buffer) < min_y_decs:
                        y_log_buffer = 0.5 * (min_y_decs - (np.log10(y_max) - np.log10(y_min)))
                    plt.ylim([10 ** (np.log10(y_min) - y_log_buffer), 10 ** (np.log10(y_max) + y_log_buffer)])
            else:
                plt.ylim(y_limits)

            return fig, ax


        def fancy_3D_plot(
                # Required data
                xdata_lists, ydata_lists, zdata_lists,

                # Optional data
                plot_styles=None, data_labels=[],
                # Standard basic settings (optional)
                figi=1, title_str='', x_label_str='x-axis', y_label_str='y-axis', z_label_str='z-axis',
                x_limits=[], y_limits=[], z_limits=[], use_mpl_limits=True, x_scale='linear', y_scale='linear',
                z_scale='linear',
                OoB_z_handling='NaN',

                fig_width_inch=9.5, fig_height_inch=6.5, title_fs=16, axis_fs=14,
                f_family='sans-serif', f_style='normal', f_variant='normal', f_weight='normal',

                fig=None, ax=None, spnrows=1, spncols=1, spindex=1,
                man_sp_placement=False, spx0=0.1, spy0=0.1, sph0=0.4, spw0=0.4,

                color='#FDFEFC', cmap='viridis', facecolors=None, depthshade=True, linestyle='-', linewidth=1,
                marker='.', markersize=5, markerfacecolor=None, markeredgecolor=None, markeredgewidth=None,
                rstride=1, cstride=1, rcount=50, ccount=50,
                alpha=None,

                # Color map options
                x_meaning='mid', y_meaning='mid',
                cbar_fs=None, cbar_size=5, cbar_pad=0.1,

                # Legend settings
                legend_position='outside bottom', legend_anchor=None, legend_ncol=1, legend_alpha=None,
        ):
            '''
            Description:
                Generate a 3D plot containing an arbitrary number of datasets.  The z-axis of each of the datasets can either be a 1-D list (describing
                   scatter points or a line) or a 2-D NumPy array (describing a surface); the x and y axes must be 1-D and match the correct dimension
                   of the z-axis dataset.

            Dependencies:
              - `import numpy as np`
              - `import matplotlib.pyplot as plt`
              - `from mpl_toolkits.mplot3d import Axes3D`
              - `import matplotlib.ticker as mticker`
              - `from mpl_toolkits.mplot3d.axis3d import Axis`
              - `import matplotlib.projections as proj`
              - `from matplotlib.colors import colorConverter`

            Inputs:
               (Required)

              - `xdata_lists` = a list containing lists/arrays of 1-D x data (or single list of xdata applied to all zdata in `z_data_lists`)
              - `ydata_lists` = a list containing lists/arrays of 1-D y data (or single list of ydata applied to all zdata in `z_data_lists`)
              - `zdata_lists` = a list containing lists/arrays of z datasets (or a single list/array),
                   - individual z datasets can be provided in either of two different formats:
                       - 1) 1-D lists (of the same dimension of the corresponding x and y lists)
                       - 2) 2-D NumPy arrays whose whidth and height match the dimensions of the x and y lists.

            Inputs:
               (Optional, basic, generic)

              - `plot_styles` = list of (or individual) strings denoting the plot style to be used for each dataset. Options include:
                  + 1-D `['line','scatter','trisurface']`   (D=`'line'`)
                  + 2-D `['surface','wireframe','trisurface','contour','filledcontour']`   (D=`'trisurface'`)
                  + 2-D_colormaps `['map_pcolormesh','map_filledcontour','map_contour']`
              - `data_labels` = a list of strings to be used as data labels in the legend (D=`[]`, no legend generated) (labels do not work for contours)
              - `figi` = figure index (D=`1`)
              - `title_str` = string to be used as the title of the plot (D=`''`)
              - `x_label_str` = string to be used as x-axis title (D=`'x-axis'`)
              - `y_label_str` = string to be used as y-axis title (D=`'y-axis'`)
              - `z_label_str` = string to be used as z-axis title (D=`'z-axis'`)
              - `x_limits` = length 2 list specifying minimum and maximum x-axis bounds [xmin,xmax] (D=auto-calculated based on `x_data_lists`)
              - `y_limits` = length 2 list specifying minimum and maximum y-axis bounds [ymin,ymax] (D=auto-calculated based on `y_data_lists`)
              - `z_limits` = length 2 list specifying minimum and maximum z-axis bounds [zmin,zmax] (D=auto-calculated based on `z_data_lists`)
              - `use_mpl_limits` = boolean specifying if Matplotlib default (`True`) or specially calculated (`False`) log-scale axis limits are used when they aren't specified (D=`True`)
              - `x_scale` = x-axis scale, either `"linear"` or `"log"`
              - `y_scale` = y-axis scale, either `"linear"` or `"log"`
              - `z_scale` = z-axis scale, either `"linear"` or `"log"`
              - `OoB_z_handling` = string denoting how z values outside of z_limits, if provided, will be handled. (D=`'NaN'`, other option are `'limits'` and `'None'`)
                               By default, out of bounds (OoB) values are replaced with `NaN`.  If this is set to `'limits'`, OoB values are set equal to the lower/upper limit provided.
                               If `'None'`, no values are replaced (this may cause errors with 3-D plots).
              - `fig_width_inch` = figure width in inches (D=`9.5`)
              - `fig_height_inch` = figure height in inches (D=`6.5`)
              - `title_fs` = title font size (D=`16`)
              - `axis_fs` = axis label font size (D=`14`)
              - `f_family` = string specifying font family (D=`'sans-serif'`); options include: `['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']`
              - `f_style` = string specifying font style (D=`'normal'`); options include: `['normal', 'italic', 'oblique']`
              - `f_variant` = string specifying font variant (D=`'normal'`); options include: `['normal', 'small-caps']`
              - `f_weight` = string specifying font weight (D=`'normal'`); options include: `['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black']`

            Inputs:
               (Optional, basic, 3D plot type specific)

                The below options are only applicable to specific `plot_styles`; this is denoted by the leftmost column.

              - `L P S W T C F` (L=`'line'`, P=`'scatter'`(points), S=`'surface'`, W=`'wireframe'`, T=`'trisurface'`, C=`'contour'`, F=`'filledcontour'`; `'o'` indicates options used by each plot type)
              - `o o o o o o o` |  `alpha` = list of (or individual) int/float of the alpha/opacity of each point/curve/surface/etc. (D=`None`)
              - `o o o o o o o` |  `color` = list of color strings to be used of same length as z_data_lists (or individual color string) (D=Matplotlib default color cycle)
              - `. . o . o o o` |  `cmap` = list of colormaps to be used of same length as z_data_lists, this will overwrite 'color' (or individual colormap) (D=`None`)
              - `o . . o . o .` |  `linestyle` = list of (or individual) strings denoting linestyle: `''`, `'-'`, `'--'`, `'-.'`, or `':'` (D=`'-'`)
              - `o . . o . o .` |  `linewidth` = list of (or individual) int/float of the width of line (D=`1`)
              - `o o . . . . .` |  `marker` = list of (or individual) marker styles (D=`'.'`) For all options, see: https://matplotlib.org/3.1.0/api/markers_api.html
              - `o o . . . . .` |  `markersize` = list of (or individual) int/float of marker size (D=`5`)
              - `o o . . . . .` |  `markerfacecolor` = list of (or individual) marker face colors (D=`None`, use value of `'color'`)
              - `o o . . . . .` |  `markeredgecolor` = list of (or individual) marker edge colors (D=`None`, use value of `'color'`)
              - `o o . . . . .` |  `markeredgewidth` = list of (or individual) int/float of marker edge widths (D=`None`)
              - `. o . . . . .` |  `depthshade` = list of (or individual) booleans to enable/disable marker shading for appearance of depth (D=`True`)
              - `. . o o . . .` |  `rstride` = [DEPRECATED] list of (or individual) int/float of array row strides (D=`1`)
              - `. . o o . . .` |  `cstride` = [DEPRECATED] list of (or individual) int/float of array column strides (D=`1`)
              - `. . o o . . .` |  `rcount`  = list of (or individual) int/float of maximum number of rows used (D=`50`)
              - `. . o o . . .` |  `ccount`  = list of (or individual) int/float of maximum number of columns used (D=`50`)
              - `. . o . . . .` |  `facecolors` = list of (or individual) array mapping colors to each facet of the zdata (D=`None`) overwrites cmap

            Inputs:
               (Optional, 2D colormap type specific)

              - `x_meaning` = string specifying if x values describe x min, x max, or central x value for each corresponding z (D=`'mid'`); options include: `['min','max','mid']` (think of like bin min/max/mid)
              - `y_meaning` = string specifying if y values describe y min, y max, or central y value for each corresponding z (D=`'mid'`); options include: `['min','max','mid']`
              - `cbar_fs` = color bar label font size (D=`axis_fs` (D=`14`))
              - `cbar_size` = color bar size expressed as an integer/float between 0 and 100 (D=`5`) (think of as percentage width)
              - `cbar_pad` = color bar padding (should be between 0 and 1) (D=`0.1`)


            Inputs:
               (Optional, advanced)

               Subplots

                 - `fig` = figure handles from existing figure to draw on (D=`None`, `fig=None` should always be used for initial subplot unless a figure canvas has already been generated)
                 - `ax` = axis handles from an existing figure to draw on (D=`None`, `ax=None` should always be used for initial subplot)
                 - `spnrows` = number of rows in final subplot (D=`1`)
                 - `spncols` = number of columns in final subplot (D=`1`)
                 - `spindex` = index of current subplot (between 1 and spnrows*spncols) (D=`1`)
                 - `man_sp_placement` = logical variable controlling manual sizing/placement of subplots using below variables (D=`False`, use automatic sizing)
                 - `spx0` = distance from canvas left edge where this plotting area should begin (D=`0.1`), generally a number around 0~1
                 - `spy0` = distance from canvas bottom edge where this plotting area should begin (D=`0.1`), generally a number around 0~1
                 - `spw0` = width of this plotting area on the canvas (D=`0.4`), generally a number around 0~1
                 - `sph0` = height of this plotting area on the canvas (D=`0.4`), generally a number around 0~1

               Legend settings

                 - `legend_position` = one of the default matplotlib legend position strings (`'best'`,`'upper right'`,`'lower center'`,`'lower left'`,etc.) to place the legend inside the plot or
                                   `'outside right'` or `'outside bottom'` to place the legend outside of the plot area (D=`'outside bottom'`, if legend is to be used)
                 - `legend_anchor` = legend anchor position (x=left-right position, y=bottom-top position) only used when legend position is set to one of the "outside" options
                                   (D=`None` which becomes `(1.0,0.75)` if position is `'outside right'` or `(0.5,-0.17)` if position is `'outside bottom'`)
                                   Note that only one coordinate usually should be adjusted.  If using an `'outside right'` legend, only the y-coordinate needs to be manipulated
                                   to scoot the legend up/down.  Likewise, for `'outside bottom'` legends only the x-coordinate needs adjusting for tuning left/right position
                 - `legend_ncol` = number of columns in legend (D=`1` in all cases except for legend_position=`'outside bottom'` where D=`len(ydata_lists)`)
                 - `legend_alpha` = alpha of legend background (D=`None`, auto determined by matplotlib)


            '''
            '''
                Notes: 
    
                Leftover string not yet used for 2D colormap specific options: 
                The below options are only applicable to specific plot styles; this is denoted by the leftmost column.
                P C F (P='map_pcolormesh', C='map_contour', F='map_filledcontour'; 'o' indicates options used by each plot type)
            '''

            # This allows for implementation of hacked in minor gridlines
            class axis3d_custom(Axis):  # https://stackoverflow.com/questions/31684448/how-to-color-a-specific-gridline-tickline-in-3d-matplotlib-scatter-plot-figure
                def __init__(self, adir, v_intervalx, d_intervalx, axes, *args, **kwargs):
                    Axis.__init__(self, adir, v_intervalx, d_intervalx, axes, *args, **kwargs)
                    self.gridline_colors = []

                def set_gridline_color(self, *gridline_info):
                    '''Gridline_info is a tuple containing the value of the gridline to change
                    and the color to change it to. A list of tuples may be used with the * operator.'''
                    self.gridline_colors.extend(gridline_info)

                def draw(self, renderer):
                    # filter locations here so that no extra grid lines are drawn
                    Axis.draw(self, renderer)
                    which_gridlines = []
                    if self.gridline_colors:
                        locmin, locmax = self.get_view_interval()
                        if locmin > locmax:
                            locmin, locmax = locmax, locmin

                        # Rudimentary clipping
                        majorLocs = [loc for loc in self.major.locator() if
                                     locmin <= loc <= locmax]
                        for i, val in enumerate(majorLocs):
                            for colored_val, color in self.gridline_colors:
                                if val == colored_val:
                                    which_gridlines.append((i, color))
                        colors = self.gridlines.get_colors()
                        for val, color in which_gridlines:
                            colors[val] = colorConverter.to_rgba(color)
                        self.gridlines.set_color(colors)
                        self.gridlines.draw(renderer, project=True)

            class XAxis(axis3d_custom):
                def get_data_interval(self):
                    'return the Interval instance for this axis data limits'
                    return self.axes.xy_dataLim.intervalx

            class YAxis(axis3d_custom):
                def get_data_interval(self):
                    'return the Interval instance for this axis data limits'
                    return self.axes.xy_dataLim.intervaly

            class ZAxis(axis3d_custom):
                def get_data_interval(self):
                    'return the Interval instance for this axis data limits'
                    return self.axes.zz_dataLim.intervalx

            class Axes3D_custom(Axes3D):
                """
                3D axes object.
                """
                name = '3d_custom'

                def _init_axis(self):
                    '''Init 3D axes; overrides creation of regular X/Y axes'''
                    self.w_xaxis = XAxis('x', self.xy_viewLim.intervalx,
                                         self.xy_dataLim.intervalx, self)
                    self.xaxis = self.w_xaxis
                    self.w_yaxis = YAxis('y', self.xy_viewLim.intervaly,
                                         self.xy_dataLim.intervaly, self)
                    self.yaxis = self.w_yaxis
                    self.w_zaxis = ZAxis('z', self.zz_viewLim.intervalx,
                                         self.zz_dataLim.intervalx, self)
                    self.zaxis = self.w_zaxis

                    for ax in self.xaxis, self.yaxis, self.zaxis:
                        ax.init3d()

            proj.projection_registry.register(Axes3D_custom)

            use_custom_3d_axis_class = False  # custom axes broken in newer version of matplotlib?

            valid_plot_styles = ['line', 'scatter', 'surface', 'wireframe', 'trisurface', 'contour', 'filledcontour',
                                 'map_pcolormesh', 'map_filledcontour', 'map_contour']
            pls_by_dims = [['line', 'scatter', 'trisurface'],
                           ['surface', 'wireframe', 'trisurface', 'contour', 'filledcontour', 'map_pcolormesh',
                            'map_filledcontour', 'map_contour']]
            pls_maps = ['map_pcolormesh', 'map_filledcontour', 'map_contour']

            if data_labels:
                include_legend = True  # used to toggle legend on/off
            else:
                include_legend = False
            single_dataset = False  # Assume multiple datasets entered, but this can be tested to see if it is the case or not.

            if (not xdata_lists) and (not ydata_lists):
                print('Warning: Both xdata and ydata lists are empty (figure index = {}, titled "{}")'.format(figi,
                                                                                                              title_str))
                single_dataset = True
                include_legend = False
                xdata_lists = [[]]
                ydata_lists = [[]]
            elif (not xdata_lists):
                print('Warning: xdata list is empty (figure index = {}, titled "{}")'.format(figi, title_str))
            elif (not ydata_lists):
                print('Warning: ydata list is empty (figure index = {}, titled "{}")'.format(figi, title_str))

            # First, determine the number of datasets which have been provided
            # This is solely determined from the z axis entries.
            # A z list consisting of only floats/ints is interpreted only as being coordinates to corresponding
            # x and y lists of the same length.
            # If wanting to generate a plot with one axis whose values are unchanging in each dataset, please make that x or y.

            # The Z list can either be an individual an item or a list of supported structures.
            # Acceptable entries for Z are:
            #  - list of values to plot a line or scatter
            #  - 2D array (numpy, not a list of lists) whose shape matches the lengths of corresponding 1D arrays of x and y data

            if isinstance(zdata_lists, list):
                if (
                all(isinstance(el, (int, float)) for el in zdata_lists)):  # just a single list of z coordinates provided
                    ndatasets = 1
                    zdata_lists = [zdata_lists]
                elif len(
                        zdata_lists) == 1:  # provided just a single dataset which could either be a list of values or 2D numpy array
                    ndatasets = 1
                else:  # provided a number of datasets which could be composed of lists of values and/or 2D numpy arrays
                    ndatasets = len(zdata_lists)
            elif isinstance(zdata_lists, np.ndarray):
                if len(np.shape(zdata_lists)) == 1:  # single 1D array
                    ndatasets = 1
                    zdata_lists = zdata_lists.tolist()
                elif len(np.shape(zdata_lists)) == 2:  # single 2D array
                    ndatasets = 1
                    zdata_lists = [zdata_lists]
                elif len(np.shape(zdata_lists)) == 3:  # divide 3D array into multiple 2D slices
                    ndatasets = np.shape(zdata_lists)[2]
                    original_zdata_lists = zdata_lists
                    zdata_lists = []
                    zdata_lists = [original_zdata_lists[:, :, i] for i in range(np.shape(original_zdata_lists)[2])]
                else:
                    print('Dimensions of zdata_lists numpy array is incorrect')
                    return 0
            else:
                print('zdata_lists is invalid.  Please enter either a list (of lists) or numpy array')
                return 0

            if ndatasets > 1:
                if isinstance(plot_styles, list):
                    for i in plot_styles:
                        if i in pls_maps:
                            print('Only 1 dataset is allowed per call of fancy_3D_plot when a map plot style is selected.')
                            return 0
                else:
                    if plot_styles in pls_maps:
                        print('Only 1 dataset is allowed per call of fancy_3D_plot when a map plot style is selected.')
                        return 0

            # Determine if 2D color map or 3D plot
            plot_2D_map = False
            plot_pcolormesh = False
            if isinstance(plot_styles, list):
                if plot_styles[0] in pls_maps:
                    plot_2D_map = True
                    if plot_styles[0] == 'map_pcolormesh': plot_pcolormesh = True
            else:
                if plot_styles in pls_maps:
                    plot_2D_map = True
                    if plot_styles == 'map_pcolormesh': plot_pcolormesh = True

            zlen = ndatasets

            # At this point, zdata_lists if just a list containing either 1D lists or 2D numpy arrays
            # For each z dataset, determine if a list of z coords (1D) or xy map (2D) array
            nzdims = []
            for i in range(len(zdata_lists)):
                nzdims.append(len(np.shape(np.array(zdata_lists[i]))))

            # Now determine how the provided x and y data map onto the provided z data

            # Check if either x or y data lists are lists of floats/ints rather than a list of lists
            xdata_only_vals = (all(isinstance(el, (int, float)) for el in xdata_lists))
            ydata_only_vals = (all(isinstance(el, (int, float)) for el in ydata_lists))

            if xdata_only_vals:
                xdata_lists = [xdata_lists for i in range(ndatasets)]
            if ydata_only_vals:
                ydata_lists = [ydata_lists for i in range(ndatasets)]

            xlen = len(xdata_lists)
            ylen = len(ydata_lists)

            # Check that all dimensions fit correctly and are consistent
            for i in range(ndatasets):
                xvals = xdata_lists[i]
                yvals = ydata_lists[i]
                zvals = zdata_lists[i]

                if nzdims[i] == 1:  # 1D list
                    zlength = len(zvals)
                    zwidth = 1
                else:  # 2D array
                    zlength = np.shape(zvals)[0]
                    zwidth = np.shape(zvals)[1]

                if zwidth == 1:  # points / line
                    if not (len(xvals) == zlength and len(yvals) == zlength):
                        print(
                            'Dimension mismatch of dataset i={} with x-length={}, y-length={}, and z-length={}, aborting.'.format(
                                str(i), str(len(xvals)), str(len(yvals)), str(zlength)))
                        return 0
                else:  # surface
                    if not (len(xvals) == zlength and len(yvals) == zwidth):  # if not fitting expected dimensions
                        if ((len(yvals) == zlength and len(xvals) == zwidth) or (plot_pcolormesh and (
                                len(yvals) == zlength + 1 or len(xvals) == zwidth + 1))):  # z vals need to be transposed
                            zdata_lists[i] = zdata_lists[i].T
                            print(
                                'Warning: Transpozing Z dataset i={} with x-length={}, y-length={}, and original z-shape from {} to {}.'.format(
                                    str(i), str(len(xvals)), str(len(yvals)), str(np.shape(zvals)),
                                    str(np.shape(zdata_lists[i]))))
                        elif plot_pcolormesh and (len(yvals) == zwidth + 1 or len(xvals) == zlength + 1):
                            print(
                                'Note: For Z dataset i={} with x-length={}, y-length={}, and original z-shape from {} to {} can only be used with map_pcolormesh due to shape.'.format(
                                    str(i), str(len(xvals)), str(len(yvals)), str(np.shape(zvals)),
                                    str(np.shape(zdata_lists[i]))))
                        else:
                            print(
                                'Dimension mismatch of dataset i={} with x-length={}, y-length={}, and z-shape={}, aborting.'.format(
                                    str(i), str(len(xvals)), str(len(yvals)), str(zlength)))
                            return 0

            fst = title_fs  # 16
            fs = axis_fs  # 14
            z_min = 1.0e10  # later used to set z-axis minimum
            z_max = 1.0e-14  # later used to set z-axis maximum
            y_min = 1.0e10  # later used to set y-axis minimum
            y_max = 1.0e-14  # later used to set y-axis maximum
            x_min = 1.0e5  # later used to set x-axis minimum
            x_max = 1.0e1  # later used to set x-axis maximum

            if z_scale == 'log':
                z_min = np.log10(z_min)
                z_max = np.log10(z_max)

            if y_scale == 'log':
                y_min = np.log10(y_min)
                y_max = np.log10(y_max)

            if x_scale == 'log':
                x_min = np.log10(x_min)
                x_max = np.log10(x_max)

            plt.rc('font', family=f_family, style=f_style, variant=f_variant, weight=f_weight)

            if fig == None:
                fig = plt.figure(figi)

            if plot_2D_map:
                ax = fig.add_subplot(spnrows, spncols, spindex)
            elif use_custom_3d_axis_class:
                ax = fig.add_subplot(spnrows, spncols, spindex, projection='3d_custom')
            else:
                ax = fig.add_subplot(spnrows, spncols, spindex, projection='3d')

            # bg_color = '#FFFFFF' #'#E1E4E6'
            # fig.patch.set_facecolor(bg_color)
            # fig.patch.set_alpha(1.0)
            # ax = plt.subplot(spnrows, spncols, spindex)

            for i in range(ndatasets):

                if include_legend:
                    label_str = data_labels[i]
                else:
                    label_str = ''

                if isinstance(plot_styles, list):
                    pls = plot_styles[i]
                elif plot_styles == None:
                    if nzdims[i] == 1:
                        pls = 'line'
                    else:
                        pls = 'trisurface'
                else:
                    pls = plot_styles
                    if pls not in valid_plot_styles:
                        print(
                            'Submitted plot style "{}" for index {} dataset is not a valid entry.  Valid options include: '.format(
                                pls, str(i)), valid_plot_styles, "Aborting.")
                        return 0
                    elif pls not in pls_by_dims[int(nzdims[i] - 1)]:
                        print(
                            'Submitted plot style "{}" for index {} dataset is not a valid entry for a {}-D dataset.  Valid options include: '.format(
                                pls, str(i), str(nzdims[i])), pls_by_dims[int(nzdims[i] - 1)], 'Aborting.')
                        return 0

                # Get settings which may be constant or vary by dataset (lists)
                if isinstance(color, list):
                    c = color[i]
                else:
                    c = color
                if c == '#FDFEFC': c = None

                if isinstance(cmap, list):
                    cmp = cmap[i]
                else:
                    cmp = cmap
                if isinstance(cmap, str): cmp = plt.get_cmap(cmp)

                if isinstance(linestyle, list):
                    ls = linestyle[i]
                else:
                    ls = linestyle
                if isinstance(linewidth, list):
                    lw = linewidth[i]
                else:
                    lw = linewidth
                if isinstance(marker, list):
                    mkr = marker[i]
                else:
                    mkr = marker
                if isinstance(markersize, list):
                    mks = markersize[i]
                else:
                    mks = markersize
                if isinstance(markerfacecolor, list):
                    mfc = markerfacecolor[i]
                else:
                    mfc = markerfacecolor
                if isinstance(markeredgecolor, list):
                    mec = markeredgecolor[i]
                else:
                    mec = markeredgecolor
                if isinstance(markeredgewidth, list):
                    mew = markeredgewidth[i]
                else:
                    mew = markeredgewidth
                if isinstance(depthshade, list):
                    depthshade_i = depthshade[i]
                else:
                    depthshade_i = depthshade

                if isinstance(rstride, list):
                    rstride_i = rstride[i]
                else:
                    rstride_i = rstride
                if isinstance(cstride, list):
                    cstride_i = cstride[i]
                else:
                    cstride_i = cstride
                if isinstance(rcount, list):
                    rcount_i = rcount[i]
                else:
                    rcount_i = rcount
                if isinstance(ccount, list):
                    ccount_i = ccount[i]
                else:
                    ccount_i = ccount
                if isinstance(facecolors, list):
                    facecolors_i = facecolors[i]
                else:
                    facecolors_i = facecolors
                if isinstance(alpha, list):
                    alpha_i = alpha[i]
                else:
                    alpha_i = alpha

                # Make actual plot

                xvals = np.array(xdata_lists[i]).astype(float)
                yvals = np.array(ydata_lists[i]).astype(float)
                zvals = np.array(zdata_lists[i]).astype(float)

                # If user provided axis bounds, enforce them now
                if x_limits:
                    if x_limits[0]: xvals[xvals < x_limits[0]] = np.NaN
                    if x_limits[1]: xvals[xvals > x_limits[1]] = np.NaN
                if y_limits:
                    if y_limits[0]: yvals[yvals < y_limits[0]] = np.NaN
                    if y_limits[1]: yvals[yvals > y_limits[1]] = np.NaN
                if z_limits:
                    if OoB_z_handling == 'NaN':
                        if z_limits[0]: zvals[zvals < z_limits[0]] = np.NaN
                        if z_limits[1]: zvals[zvals > z_limits[1]] = np.NaN
                    elif OoB_z_handling == 'limits':
                        if z_limits[0]: zvals[zvals < z_limits[0]] = z_limits[0]
                        if z_limits[1]: zvals[zvals > z_limits[1]] = z_limits[1]

                if z_scale == 'log':
                    zvals[(zvals <= 0)] = np.NaN
                    zvals = np.log10(zvals)
                if y_scale == 'log':
                    yvals[yvals <= 0] = np.NaN
                    yvals = np.log10(yvals)
                if x_scale == 'log':
                    xvals[xvals <= 0] = np.NaN
                    xvals = np.log10(xvals)

                if len(yvals) != 0:
                    if len(yvals[np.nonzero(yvals)]) != 0:
                        if min(yvals[np.nonzero(yvals)]) < y_min: y_min = min(yvals[np.nonzero(yvals)])
                        # if min(yvals)<y_min: y_min = min(yvals)
                        if np.nanmax(yvals) > y_max: y_max = np.nanmax(yvals)
                        if np.nanmin(xvals) < x_min: x_min = np.nanmin(xvals)
                        if np.nanmax(xvals) > x_max: x_max = np.nanmax(xvals)
                        if np.nanmin(zvals) < z_min: z_min = np.nanmin(zvals)
                        if np.nanmax(zvals) > z_max: z_max = np.nanmax(zvals)

                if nzdims[i] == 1:  # 1D list
                    zlength = len(zvals)
                    zwidth = 1
                else:  # 2D array
                    zlength = np.shape(zvals)[0]
                    zwidth = np.shape(zvals)[1]

                # Plotting functions

                if nzdims[i] == 1:
                    # line
                    if pls == 'line':
                        ax.plot(xvals, yvals, zvals, label=label_str,
                                color=c, linestyle=ls, linewidth=lw, alpha=alpha_i,
                                marker=mkr, markersize=mks, markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew)

                    # scatter
                    elif pls == 'scatter':
                        if not c: c = mfc  # if no color defined, check to see if marker face color was defined
                        ax.scatter(xvals, yvals, zvals, label=label_str,
                                   color=c, depthshade=depthshade_i, alpha=alpha_i,
                                   marker=mkr, s=mks ** 2, linewidths=mew, edgecolors=mec)

                    # trisurface
                    elif pls == 'trisurface':
                        if cmp != None:
                            c = None
                        if facecolors != None:
                            c = None
                            cmap = None
                        ps1 = ax.plot_trisurf(xvals, yvals, zvals, label=label_str,
                                              color=c, cmap=cmp, facecolors=facecolors_i, alpha=alpha_i)
                        ps1._facecolors2d = ps1._facecolors3d
                        ps1._edgecolors2d = ps1._edgecolors3d

                    else:
                        print(
                            'Encountered incompatability with plot style {} and data dimensionality {} for data index {}.  Aborting.'.format(
                                pls, str(nzdims[i]), str(i)))

                else:
                    xvals_original, yvals_original = xvals, yvals
                    xvals, yvals = np.meshgrid(xvals, yvals)

                    # surface
                    if pls == 'surface':
                        if cmp != None:
                            c = None
                        if facecolors != None:
                            c = None
                            cmap = None
                        ps1 = ax.plot_surface(xvals, yvals, zvals.T, label=label_str,
                                              color=c, cmap=cmp, facecolors=facecolors_i, alpha=alpha_i,
                                              rcount=rcount_i, ccount=ccount_i,
                                              antialiased=False,
                                              vmin=z_min, vmax=z_max)  # this line was once not needed
                        ps1._facecolors2d = ps1._facecolor3d
                        ps1._edgecolors2d = ps1._edgecolor3d

                    # wireframe
                    elif pls == 'wireframe':
                        ax.plot_wireframe(xvals, yvals, zvals.T, label=label_str,
                                          color=c, linestyle=ls, linewidth=lw, alpha=alpha_i,
                                          rcount=rcount_i, ccount=ccount_i)

                    elif pls == 'trisurface':
                        xvals = np.reshape(xvals, -1)
                        yvals = np.reshape(yvals, -1)
                        xtri = []
                        ytri = []
                        ztri = []
                        for yi in range(np.shape(zvals)[1]):
                            for xi in range(np.shape(zvals)[0]):
                                ztri.append(zvals[xi, yi])
                                # xtri.append(xdata_lists[i][xi])
                                # ytri.append(ydata_lists[i][yi])

                        if cmp != None:
                            c = None
                        if facecolors != None:
                            c = None
                            cmap = None

                        ps1 = ax.plot_trisurf(xvals, yvals, ztri, label=label_str,
                                              color=c, cmap=cmp, facecolors=facecolors_i, alpha=alpha_i)
                        ps1._facecolors2d = ps1._facecolors3d
                        ps1._edgecolors2d = ps1._edgecolors3d

                    # contour
                    elif pls == 'contour':
                        if cmp != None: c = None
                        ax.contour(xvals, yvals, zvals.T,
                                   colors=c, cmap=cmp, linestyles=ls, linewidths=lw, alpha=alpha_i)

                    # filled contour
                    elif pls == 'filledcontour':
                        if cmp != None: c = None
                        ax.contourf(xvals, yvals, zvals.T,
                                    colors=c, cmap=cmp, alpha=alpha_i)



                    # map contour
                    elif pls == 'map_pcolormesh':
                        if cmp != None: c = None
                        # first, check if x and y dims are 1 larger than z dims
                        expand_x, expand_y = False, False
                        if np.shape(yvals)[0] == np.shape(zvals)[1]:
                            expand_y = True
                        if np.shape(xvals)[1] == np.shape(zvals)[0]:
                            expand_x = True
                        # make x any y bigger by 1 since pcolormesh takes all edges, not just midpoints
                        if expand_x:
                            dx = xvals[:, 1:] - xvals[:, :-1]
                            if x_meaning == 'min':
                                newx = (xvals[:, -1] + dx[:, -1]).reshape(len(xvals[:, 0]), 1)
                                xvals = np.hstack((xvals, newx))
                                yvals = np.hstack((yvals, yvals[:, -1].reshape(len(yvals[:, -1]), 1)))
                            if x_meaning == 'max':
                                newx = (xvals[:, 0] - dx[:, 0]).reshape(len(xvals[:, 0]), 1)
                                xvals = np.hstack((newx, xvals))
                                yvals = np.hstack((yvals[:, 0].reshape(len(yvals[:, 0]), 1), yvals))
                            if x_meaning == 'mid':
                                newx = (xvals[:, 0] - 0.5 * dx[:, 0]).reshape(len(xvals[:, 0]), 1)
                                dx = np.hstack((dx, np.tile(dx[:, [-1]], 1)))
                                xvals = xvals + 0.5 * dx
                                xvals = np.hstack((newx, xvals))
                                yvals = np.hstack((yvals[:, 0].reshape(len(yvals[:, 0]), 1), yvals))
                        if expand_y:
                            dy = yvals[1:] - yvals[:-1]
                            if y_meaning == 'min':
                                newy = yvals[-1, :] + dy[-1, :]
                                yvals = np.vstack((yvals, newy))
                                xvals = np.vstack((xvals, xvals[-1, :]))
                            if y_meaning == 'max':
                                newy = yvals[0, :] - dy[0, :]
                                yvals = np.vstack((newy, yvals))
                                xvals = np.vstack((xvals[0, :], xvals))
                            if y_meaning == 'mid':
                                newy = yvals[0, :] - 0.5 * dy[0, :]
                                dy = np.vstack((dy, np.tile(dy[[-1], :], 1)))
                                yvals = yvals + 0.5 * dy
                                yvals = np.vstack((newy, yvals))
                                xvals = np.vstack((xvals[0, :], xvals))

                        pcm2d = ax.pcolormesh(xvals, yvals, zvals.T,
                                              cmap=cmp, linestyles=ls, linewidths=lw, alpha=alpha_i)


                    # map contour (normal or filled)
                    elif pls == 'map_filledcontour' or pls == 'map_contour':
                        if cmp != None: c = None
                        if x_meaning != 'mid':
                            # shift x values to be midpoints
                            dx = xvals[:, 1:] - xvals[:, :-1]
                            if x_meaning == 'min':
                                dx = np.hstack((dx, np.tile(dx[:, [-1]], 1)))
                                xvals = xvals + 0.5 * dx
                            if x_meaning == 'max':
                                dx = np.hstack((np.tile(dx[:, [0]], 1), dx))
                                xvals = xvals - 0.5 * dx
                        if y_meaning != 'mid':
                            # shift y values to be midpoints
                            dy = yvals[1:] - yvals[:-1]
                            if y_meaning == 'min':
                                dy = np.vstack((dy, np.tile(dy[[-1], :], 1)))
                                yvals = yvals + 0.5 * dy
                            if y_meaning == 'max':
                                dy = np.vstack((np.tile(dy[[0], :], 1), dy))
                                yvals = yvals - 0.5 * dy
                                # print(yvals)

                        if pls == 'map_contour':
                            pcm2d = ax.contour(xvals, yvals, zvals.T,
                                               cmap=cmp, linestyles=ls, alpha=alpha_i)
                        else:
                            pcm2d = ax.contourf(xvals, yvals, zvals.T,
                                                cmap=cmp, linestyles=ls, alpha=alpha_i)

                    else:
                        print(
                            'Encountered incompatability with plot style {} and data dimensionality {} for data index {}.  Aborting.'.format(
                                pls, str(nzdims[i]), str(i)))

            if title_str.strip() != '':
                window_title = slugify(title_str)  # "comparison_fig"
            else:
                window_title = 'Figure ' + str(figi)
            # window_title = window_title.replace('b','',1) # remove leading 'b' character from slugify process
            fig.canvas.manager.set_window_title(window_title)

            # hangle figure/legend positioning/sizing
            # First, figure size
            default_fig_x_in = fig_width_inch
            default_fig_y_in = fig_height_inch
            fig_x_in = default_fig_x_in
            fig_y_in = default_fig_y_in
            fig.set_size_inches(fig_x_in, fig_y_in)

            mpl_leg_pos_names = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                                 'center right', 'lower center', 'upper center', 'center']
            custom_leg_pos_names = ['outside right', 'outside bottom']

            if include_legend and legend_position in custom_leg_pos_names:
                if legend_anchor == None:
                    if legend_position == 'outside right':
                        legend_anchor = (1.0, 0.75)
                    elif legend_position == 'outside bottom':
                        legend_anchor = (0.5, -0.05)
                leg1_anchor = legend_anchor  # varied items
                handles_l1, labels_l1 = ax.get_legend_handles_labels()
                if legend_position == 'outside right':
                    legend1 = ax.legend(handles_l1, labels_l1, loc='upper left', bbox_to_anchor=leg1_anchor,
                                        ncol=legend_ncol, framealpha=legend_alpha)
                elif legend_position == 'outside bottom':
                    if legend_ncol == 1 and len(data_labels) > 1: legend_ncol = len(data_labels)
                    legend1 = ax.legend(handles_l1, labels_l1, loc='upper center', bbox_to_anchor=leg1_anchor,
                                        ncol=legend_ncol, framealpha=legend_alpha)
                ax.add_artist(legend1)
                fig.canvas.draw()
                f1 = legend1.get_frame()
                l1_w0_px, l1_h0_px = f1.get_width(), f1.get_height()
                l_w0_in, l_h0_in = l1_w0_px / fig.dpi, l1_h0_px / fig.dpi  # width and height of legend, in inches
            else:
                l_w0_in, l_h0_in = 0.0, 0.0
                if include_legend and legend_position not in custom_leg_pos_names:  # use matplotlib default-style legend inside plot area
                    ax.legend(loc=legend_position, ncol=legend_ncol, framealpha=legend_alpha)

            n_title_lines = 0
            if title_str.strip() != '':
                n_title_lines = 1 + title_str.count('\n')
            n_xlabel_lines = 0
            if x_label_str.strip() != '':
                n_xlabel_lines = 1 + x_label_str.count('\n')
            n_ylabel_lines = 0
            if y_label_str.strip() != '':
                n_ylabel_lines = 1 + y_label_str.count('\n')
            n_zlabel_lines = 1
            if z_label_str.strip() != '':
                n_zlabel_lines = 1 + z_label_str.count('\n')

            if plot_2D_map:
                # These values are good, do not change them.  (derived while working on SHAEDIT project)
                # INCORPORATE WIDTH OF COLORBAR AND ITS LABEL?
                x0bar = 0.60 + 0.200 * n_ylabel_lines  # inches, horizontal space needed for ylabel
                y0bar = 0.45 + 0.200 * n_xlabel_lines  # inches, vertical space needed for xticks/numbers, xlabel and any extra lines it has
                t0bar = 0.10 + 0.300 * n_title_lines  # inches, vertical space needed for title
                del_l_in = 0.15  # inches, extra horizontal padding right of legend
            else:
                # These values are good, do not change them.  (derived while working on on this function specifically for 3D plotting)
                x0bar = 0.00 + 0.200 * (n_zlabel_lines - 1)  # inches, horizontal space needed for ylabel
                y0bar = 0.45 + 0.200 * max(n_xlabel_lines,
                                           n_ylabel_lines)  # inches, vertical space needed for xticks/numbers, xlabel and any extra lines it has
                t0bar = 0.10 + 0.300 * n_title_lines  # inches, vertical space needed for title
                del_l_in = 0.15  # inches, extra horizontal padding right of legend

            # adjust legend spacing depending on its position
            if legend_position == 'outside right':
                l_h0_in = 0.0
            elif legend_position == 'outside bottom':
                l_w0_in = 0.0

            # Plot window placement and sizing
            x0 = x0bar / fig_x_in  # distance from left edge that plot area begins
            y0 = y0bar / fig_y_in + (l_h0_in / fig_y_in)  # distance from bottom edge that plot area begins
            h0 = 1 - (y0bar + t0bar) / fig_y_in - (
                        l_h0_in / fig_y_in)  # height of plot area, set to be full height minus space needed for title, x-label, and potentially an outside bottom legend
            w0 = 1 - x0 - (l_w0_in / fig_x_in) - (
                        del_l_in / fig_x_in)  # width of plot area, set to be full width minus space needed for y-label and potentially an outside right legend

            if man_sp_placement:
                x0 = spx0
                y0 = spy0
                h0 = sph0
                w0 = spw0

            # Set size and location of the plot on the canvas
            box = ax.get_position()
            # all vals in [0,1]: left, bottom, width, height
            if not man_sp_placement and (spnrows != 1 or spncols != 1):
                pstr = 'Warning: It is highly encouraged that subplots be positioned manually.\n'
                pstr += '   This is done by setting man_sp_placement=True and then adjusting\n'
                pstr += '   the parameters spx0, spy0, sph0, and spw0 for each subplot.\n'
                pstr += '   The current plot was automatically sized by matplotlib.\n'
                print(pstr)
            else:
                ax.set_position([x0, y0, w0, h0])

            if plot_2D_map:

                ax.set_title(title_str, fontsize=fst)
                plt.xlabel(x_label_str, fontsize=fs)
                plt.ylabel(y_label_str, fontsize=fs)
                plt.xscale(x_scale)
                plt.yscale(y_scale)

                zoom_mult = 1.0
                x_log_buffer = 0.15 * zoom_mult
                y_log_buffer = 0.2 * zoom_mult
                min_x_decs = 2
                min_y_decs = 2

                x_scale = 'linear'
                if not x_limits:
                    if x_scale == 'log':  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                        if (np.log10(x_max) - np.log10(x_min) + 2 * x_log_buffer) < min_x_decs:
                            x_log_buffer = 0.5 * (min_x_decs - (np.log10(x_max) - np.log10(x_min)))
                        plt.xlim([10 ** (np.log10(x_min) - x_log_buffer), 10 ** (np.log10(x_max) + x_log_buffer)])
                else:
                    plt.xlim(x_limits)

                if not y_limits:
                    if y_scale == 'log':  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                        if (np.log10(y_max) - np.log10(y_min) + 2 * y_log_buffer) < min_y_decs:
                            y_log_buffer = 0.5 * (min_y_decs - (np.log10(y_max) - np.log10(y_min)))
                        plt.ylim([10 ** (np.log10(y_min) - y_log_buffer), 10 ** (np.log10(y_max) + y_log_buffer)])
                else:
                    plt.ylim(y_limits)

                if z_limits:
                    if z_scale == 'log':
                        zlogmin = None
                        zlogmax = None
                        if z_limits[0]: zlogmin = np.log10(z_limits[0])
                        if z_limits[1]: zlogmax = np.log10(z_limits[1])
                        pcm2d.set_clim(vmin=zlogmin, vmax=zlogmax)
                    else:
                        pcm2d.set_clim(vmin=z_limits[0], vmax=z_limits[1])

                def fmt(x, pos):
                    a, b = '{:.2e}'.format(x).split('e')
                    b = int(b)

                    if z_scale == 'log':
                        return r'$10^{{{:g}}}$'.format(x)
                    else:
                        if b < -2 or b > 3:
                            return r'${:g} \times 10^{{{}}}$'.format(np.float(a), b)
                        else:
                            return '{:g}'.format(x)

                divider = make_axes_locatable(ax)
                cbar_size_str = '{:g}'.format(cbar_size) + '%'
                cax = divider.append_axes("right", size=cbar_size_str, pad=cbar_pad)

                cbar = plt.colorbar(pcm2d, cax=cax, format=ticker.FuncFormatter(fmt))
                if cbar_fs == None:
                    cbar_fs = fs
                cbar.set_label(z_label_str, fontsize=cbar_fs)
                # cbar.solids.set_rasterized(True)
                cbar.solids.set_edgecolor("face")
                # cbar.set_alpha(alpha_i)
                # cbar.draw_all()

                ax.set_position([x0, y0, w0, h0])

            else:  # if 3D-plot
                ax.set_title(title_str, fontsize=fst)
                plt.xlabel(x_label_str, fontsize=fs)
                plt.ylabel(y_label_str, fontsize=fs)
                ax.set_zlabel(z_label_str, fontsize=fs)
                # Current matplotlib set_scale commands for log scale are borked completely beyond use, manually add log support
                # ax.set_xscale(x_scale)
                # ax.set_yscale(y_scale)
                # ax.set_zscale(z_scale)
                # plt.grid(b=True, which='major', linestyle='-', alpha=0.25) # doesn't affect 3D axis
                # plt.grid(b=True, which='minor', linestyle='-', alpha=0.10)
                # ensure at least minimum number of decades are present on a plot by increasing padding if necessary
                zoom_mult = 1.0
                x_log_buffer = 0.15 * zoom_mult
                y_log_buffer = 0.2 * zoom_mult
                z_log_buffer = 0.2 * zoom_mult
                min_x_decs = 1
                min_y_decs = 1
                min_z_decs = 1

                manually_calculate_axis_bounds = not use_mpl_limits  # if False, use default Matplotlib axis bounds; if True, use specially calculated axis bounds

                if not x_limits:
                    if x_scale == 'log' and manually_calculate_axis_bounds:  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                        if ((x_max) - (x_min) + 2 * x_log_buffer) < min_x_decs:
                            x_log_buffer = 0.5 * (min_x_decs - ((x_max) - (x_min)))
                        ax.set_xlim([((x_min) - x_log_buffer), ((x_max) + x_log_buffer)])
                else:
                    if x_scale == 'log':
                        xlimsnew = []
                        for limi in range(2):
                            if x_limits[limi]:
                                xlimsnew.append(np.log10(x_limits[limi]))
                            else:
                                xlimsnew.append(None)
                        ax.set_xlim(xlimsnew)
                    else:
                        ax.set_xlim(x_limits)

                if not y_limits:
                    if y_scale == 'log' and manually_calculate_axis_bounds:  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                        if ((y_max) - (y_min) + 2 * y_log_buffer) < min_y_decs:
                            y_log_buffer = 0.5 * (min_y_decs - ((y_max) - (y_min)))
                        ax.set_ylim([((y_min) - y_log_buffer), ((y_max) + y_log_buffer)])
                else:
                    if y_scale == 'log':
                        ylimsnew = []
                        for limi in range(2):
                            if y_limits[limi]:
                                ylimsnew.append(np.log10(y_limits[limi]))
                            else:
                                ylimsnew.append(None)
                        ax.set_ylim(ylimsnew)
                    else:
                        ax.set_ylim(y_limits)

                if not z_limits:
                    if z_scale == 'log' and manually_calculate_axis_bounds:  # use fancy code to determine bounds, otherwise, let matplotlib automatically generate boundaries
                        if ((z_max) - (z_min) + 2 * z_log_buffer) < min_z_decs:
                            z_log_buffer = 0.5 * (min_z_decs - ((z_max) - (z_min)))
                        ax.set_zlim([((z_min) - z_log_buffer), ((z_max) + z_log_buffer)])
                else:
                    if z_scale == 'log':
                        zlimsnew = []
                        for limi in range(2):
                            if z_limits[limi]:
                                zlimsnew.append(np.log10(z_limits[limi]))
                            else:
                                zlimsnew.append(None)
                        ax.set_zlim(zlimsnew)
                    else:
                        ax.set_zlim(z_limits)

                act_xmin, act_xmax = ax.get_xlim()
                act_ymin, act_ymax = ax.get_ylim()
                act_zmin, act_zmax = ax.get_zlim()

                def round_up_to_nearest_multiple(val, mult=1):
                    round_val = np.ceil(val / mult) * mult
                    if isinstance(mult, int) or (abs(round_val) % 1 < 0.01): round_val = int(round_val)
                    return round_val

                def round_down_to_nearest_multiple(val, mult=1):
                    round_val = np.floor(val / mult) * mult
                    if isinstance(mult, int) or (abs(round_val) % 1 < 0.01): round_val = int(round_val)
                    return round_val

                def get_ints_between_2_vals(vmin, vmax):
                    stepval = 1
                    # vmini, vmaxi = int(np.ceil(vmin)),int(np.floor(vmax))
                    if (vmax - vmin) <= 1:
                        stepval = 0.25
                    elif (vmax - vmin) <= 2:
                        stepval = 0.5
                    vmini = round_up_to_nearest_multiple(vmin, stepval)
                    vmaxi = round_down_to_nearest_multiple(vmax, stepval)
                    tick_list = list(np.arange(vmini, vmaxi + stepval, stepval))
                    return tick_list

                def get_log_minor_ticks_between_bounds(vmin, vmax):
                    minor_tick_list = []
                    # get powers of min and max
                    minpower = np.sign(vmin) * divmod(abs(vmin), 1)[0]  # integer portion of vmin
                    maxpower = np.sign(vmax) * divmod(abs(vmax), 1)[0]  # integer portion of vmax
                    # determine leading number in base 10
                    min_lead_digit = divmod(((10 ** vmin) / (10 ** minpower)), 1)[0] + 1
                    max_lead_digit = divmod(((10 ** vmax) / (10 ** maxpower)), 1)[0]
                    cdigit = min_lead_digit
                    cpower = minpower
                    cval = cdigit * (10 ** cpower)
                    maxval = max_lead_digit * (10 ** maxpower)
                    while cval < maxval:
                        minor_tick_list.append(np.log10(cval))
                        cdigit += 1
                        if cdigit == 10:
                            cdigit = 2
                            cpower += 1
                        cval = cdigit * (10 ** cpower)
                    return minor_tick_list

                def log_tick_formatter(vals_list):
                    tstr_list = []
                    for val in vals_list:
                        tstr = r'10$^{{{:g}}}$'.format(val)
                        tstr_list.append(tstr)
                    return tstr_list

                if z_scale == 'log':
                    zticks = get_ints_between_2_vals(act_zmin, act_zmax)
                    ztick_labs = log_tick_formatter(zticks)
                    ax.set_zticks(zticks)
                    ax.set_zticklabels(ztick_labs)

                if y_scale == 'log':
                    yticks = get_ints_between_2_vals(act_ymin, act_ymax)
                    ytick_labs = log_tick_formatter(yticks)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(ytick_labs)

                if x_scale == 'log':
                    xticks = get_ints_between_2_vals(act_xmin, act_xmax)
                    xtick_labs = log_tick_formatter(xticks)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labs)

                # ax.set_yticks(yticks+np.log10(np.array([2,3,4,5,8])).tolist())
                # yticks = yticks + np.log10(np.array([2,3,4,5,8])).tolist()

                grid_alpha = 0.25
                grid_alpha_minor = 0.05

                if x_scale == 'log':
                    ax.xaxis._axinfo["grid"]['color'] = (0, 0, 0, grid_alpha_minor)
                else:
                    ax.xaxis._axinfo["grid"]['color'] = (0, 0, 0, grid_alpha)
                if y_scale == 'log':
                    ax.yaxis._axinfo["grid"]['color'] = (0, 0, 0, grid_alpha_minor)
                else:
                    ax.yaxis._axinfo["grid"]['color'] = (0, 0, 0, grid_alpha)
                if z_scale == 'log':
                    ax.zaxis._axinfo["grid"]['color'] = (0, 0, 0, grid_alpha_minor)
                else:
                    ax.zaxis._axinfo["grid"]['color'] = (0, 0, 0, grid_alpha)

                if use_custom_3d_axis_class:
                    # If log scale, add minor grid lines
                    if x_scale == 'log':
                        xticks_minor = get_log_minor_ticks_between_bounds(act_xmin, act_xmax)
                        xgridlines = []
                        for i in range(len(xticks)):
                            xgridlines.append((xticks[i], (0, 0, 0, grid_alpha)))
                        for i in range(len(xticks_minor)):
                            xgridlines.append((xticks_minor[i], (1, 1, 1, grid_alpha_minor)))
                        ax.set_xticks(xticks + xticks_minor)
                        ax.xaxis.set_gridline_color(*xgridlines)
                    if y_scale == 'log':
                        yticks_minor = get_log_minor_ticks_between_bounds(act_ymin, act_ymax)
                        ygridlines = []
                        for i in range(len(yticks)):
                            ygridlines.append((yticks[i], (0, 0, 0, grid_alpha)))
                        for i in range(len(yticks_minor)):
                            ygridlines.append((yticks_minor[i], (0, 0, 0, grid_alpha_minor)))
                        ax.set_yticks(yticks + yticks_minor)
                        ax.yaxis.set_gridline_color(*ygridlines)
                    if z_scale == 'log':
                        zticks_minor = get_log_minor_ticks_between_bounds(act_zmin, act_zmax)
                        zgridlines = []
                        for i in range(len(zticks)):
                            zgridlines.append((zticks[i], (0, 0, 0, grid_alpha)))
                        for i in range(len(zticks_minor)):
                            zgridlines.append((zticks_minor[i], (0, 0, 0, grid_alpha_minor)))
                        ax.set_zticks(zticks + zticks_minor)
                        ax.zaxis.set_gridline_color(*zgridlines)

                # For some more info, see https://dawes.wordpress.com/2014/06/27/publication-ready-3d-figures-from-matplotlib/
                # Tick positioning
                [t.set_va('center') for t in ax.get_yticklabels()]
                [t.set_ha('center') for t in ax.get_yticklabels()]
                [t.set_va('center') for t in ax.get_xticklabels()]
                [t.set_ha('center') for t in ax.get_xticklabels()]
                [t.set_va('center') for t in ax.get_zticklabels()]
                [t.set_ha('center') for t in ax.get_zticklabels()]
                ''
                tick_infactor = 0.0
                tick_outfactor = 0.2

                # adjusts length of ticks on inside/outside of plot
                ax.xaxis._axinfo['tick']['inward_factor'] = tick_infactor
                ax.xaxis._axinfo['tick']['outward_factor'] = tick_outfactor
                ax.yaxis._axinfo['tick']['inward_factor'] = tick_infactor
                ax.yaxis._axinfo['tick']['outward_factor'] = tick_outfactor
                ax.zaxis._axinfo['tick']['inward_factor'] = tick_infactor
                ax.zaxis._axinfo['tick']['outward_factor'] = tick_outfactor

                # Background
                ax.xaxis.pane.set_edgecolor('black')
                ax.yaxis.pane.set_edgecolor('black')
                # ax.zaxis.pane.set_edgecolor('black')
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                # ax.view_init(elev=10, azim=135)

            return fig, ax


    # Timer start
    start = time.time()
    figi = 1
else:
    start_process = time.time()

'''
███████ ███████ ████████ ████████ ██ ███    ██  ██████  ███████         ██     ███████ ██ ██      ███████ ███████ 
██      ██         ██       ██    ██ ████   ██ ██       ██             ██      ██      ██ ██      ██      ██      
███████ █████      ██       ██    ██ ██ ██  ██ ██   ███ ███████       ██       █████   ██ ██      █████   ███████ 
     ██ ██         ██       ██    ██ ██  ██ ██ ██    ██      ██      ██        ██      ██ ██      ██           ██ 
███████ ███████    ██       ██    ██ ██   ████  ██████  ███████     ██         ██      ██ ███████ ███████ ███████                                                                                   
'''
# start_of_settings_section

if __name__ == '__main__':
    global source_coordinates, NOVO_face_x_coord, axis_normal_to_array_face
    global im_half_dim, pixel_vwidth, pixel_hwidth, max_En0MeV_allowed
    global make_neutron_cones, make_gamma_cones, image_neutron_events, image_gamma_events
    global fast_im_half_dim, fast_pixel_vwidth, fast_pixel_hwidth, fast_max_num_imaged_cones
    global fast_make_neutron_cones, fast_make_gamma_cones, fast_image_neutron_events, fast_image_gamma_events

    on_home_PC = False
    on_lab_PC = False

    on_Linux_machine = not sys.platform.startswith('win')
    if on_Linux_machine:
        cwd = os.getcwd() + '/'
        main_drive_base_path = cwd
        data_drive_base_path = main_drive_base_path
        #imaging_code_testing_path = main_drive_base_path + r'imaging\initial_testing\\' # used for PHITS sims
        #imaging_other_data_path = main_drive_base_path + r'imaging\root_files\\' # used for Lena's data
        if 'phits' in str(main_drive_base_path).lower():
            using_experimental_data = False
        else:
            using_experimental_data = True
    else:
        main_drive_base_path = r'G:\Cloud\OneDrive - Høgskulen på Vestlandet\work\HVL_postdoc\NOVO\\'
        if os.path.exists(main_drive_base_path):
            on_home_PC = True

        if on_home_PC:
            #main_drive_base_path = r'D:\Cloud\OneDrive - Høgskulen på Vestlandet\work\HVL_postdoc\NOVO\\'
            main_drive_base_path = r'G:\Cloud\OneDrive\work\HVL_postdoc\NOVO\\'
            # data_drive_base_path = main_drive_base_path
            #data_drive_base_path = r'F:\\'
            data_drive_base_path = r'G:\work_extra\\'
        elif on_lab_PC:
            main_drive_base_path = r'C:\\'
            data_drive_base_path = main_drive_base_path
        else:
            main_drive_base_path = r'C:\work\HVL_postdoc\NOVO\\'
            data_drive_base_path = r'D:\\'

        imaging_code_testing_path = main_drive_base_path + r'imaging\initial_testing\\'
        imaging_other_data_path = main_drive_base_path + r'imaging\root_files\\'

        using_experimental_data = True  # mark True if using ACTUAL experiment data, False if MC data
        if use_CLI:
            if "phits" in str(input_filedir).lower():
                using_experimental_data = False

    if not using_experimental_data:

        data_drive_base_path += 'phits_simulations\\'
        main_drive_base_path += 'phits_simulations\\'

        # Simulations for DT experiments in Dresden
        # sim_base_folder_name = 'parallel_2d3t'
        # sim_base_folder_name = 'parallel_3d2t'
        # sim_base_folder_name = 'parallel_3d2t_d25cm'
        # sim_base_folder_name = 'parallel_3d123t'
        # sim_base_folder_name = 'woven_3d2t'
        # sim_base_folder_name = 'woven_3d123t'
        # sim_base_folder_name = 'parallel_3d3t'
        # sim_base_folder_name = 'woven_3d2t_elastic-only'

        # sim_base_folder_name = 'parallel_3d2t_elastic_only'
        # sim_base_folder_name = 'parallel_3d2t_elastic_only_emode2'
        # sim_base_folder_name = 'parallel_3d2t_elastic_only_ielas1'
        # sim_base_folder_name = 'parallel_3d2t_nonela_only'
        # sim_base_folder_name = 'parallel_3d2t_nuclear_only'
        # sim_base_folder_name = 'parallel_3d2t_nuclear_pn_only'

        # sim_base_folder_name = 'parallel_3d2t_elastic_only_fixed'
        # sim_base_folder_name = 'parallel_3d2t_nonela_only_fixed'

        sim_base_folder_name = 'parallel_3d2t_nuclear_only_fixed'

        # sim_base_folder_name = 'parallel_3d2t_2MeV_gammas'

        # sim_base_folder_name = 'parallel_3d2t_shifted-ring-source'
        # sim_base_folder_name = 'demonstrator_design'
        # sim_base_folder_name = 'woven_3d2t'
        # sim_base_folder_name = 'emulate_Lena'

        sim_base_folder_name_only = sim_base_folder_name
        sim_base_folder_name = 'DTinDE\imaging\\' + sim_base_folder_name


        sim_base_folder_name = "src_(0,0,-200)_14.8MeV"
        #if len(sys.argv) > 2:  # allow for specification of number of workers via command line
        #    sim_base_folder_name = sys.argv[2]
        sim_base_folder_name_only = sim_base_folder_name
        sim_base_folder_name = 'mNOVOv4_PTB\shifts\\' + sim_base_folder_name

        main_drive_base_path += sim_base_folder_name + '\\'
        data_drive_base_path += sim_base_folder_name + '\\'

        n_history_power_of_10 = '8'
        # n_history_power_of_10 = '7'
        # n_history_power_of_10 = '9'

        limit_of_imaging_set_X_events_per_par = None
        # limit_of_imaging_set_X_events_per_par = 1000

        use_PT_on_tissue_mNOVOv4_settings = True
        if use_PT_on_tissue_mNOVOv4_settings:
            print('WARNING: setting use_PT_on_tissue_mNOVOvFour_settings = True is enabled!')
            if on_Linux_machine: # executing code from fwklux8 directory
                sim_base_folder_name = main_drive_base_path
                if on_home_PC:
                    drive_letter = 'G'
                else:
                    drive_letter = 'C'
                main_drive_base_path = drive_letter + ':\Cloud\OneDrive\work\HVL_researcher\EIC_WPs\WP2\PHITS\\'
                sim_base_folder_name = 'sample_mNOVO_PT_imaging\\fwklux8\\'
                sim_base_folder_name += 'unmoderated_neutrons\\'
                #sim_base_folder_name += 'tissue_cylinder\\'
                main_drive_base_path += sim_base_folder_name
            data_drive_base_path = main_drive_base_path
            n_history_power_of_10 = '8' # '9'


        run_data_folder_path_sync = main_drive_base_path + '1E{:}_histories\\'.format(n_history_power_of_10)

        if use_CLI: run_data_folder_path_sync = input_filedir

        image_data_pickle_name = 'imaging_data_records.pickle'
        if limit_of_imaging_set_X_events_per_par != None:
            image_data_pickle_name = image_data_pickle_name.replace('.pickle', '_limited_to_{:g}_events.pickle'.format(
                int(limit_of_imaging_set_X_events_per_par)))
        if use_CLI:
            image_data_pickle_path = Path.joinpath(run_data_folder_path_sync, image_data_pickle_name)
        else:
            image_data_pickle_path = run_data_folder_path_sync + image_data_pickle_name



        reject_theta_diff_thresh = 90 * (np.pi / 180)  # -0.05  # negative = fractional diff, positive = absolute diff, None = don't reject any based on angle

        bar1_IDs = []
        bar2_IDs = []


        img_tmp_folder_name = ''
        # img_tmp_folder_name = 'OGS_only\\'
        # img_tmp_folder_name = 'M600_only\\'
        # img_tmp_folder_name = 'OGS_first\\'
        # img_tmp_folder_name = 'M600_first\\'
        # img_tmp_folder_name = 'OGS_only_high-res\\'
        # img_tmp_folder_name = 'M600_first_then_OGS\\bar3_to_bar4\\'
        # img_tmp_folder_name = 'test\\'

        # img_tmp_folder_name = 'list_mode_files\\'

        # img_tmp_folder_name = 'test_parallel\\'

        #img_tmp_folder_name += input_filename.replace('.root', '')

        if use_fastmode:
            img_tmp_folder_name += '/fastmode'

        if use_CLI:
            input_filepath = str(input_path)
            #output_filepath = output_filedir
            images_path = output_filedir
            #images_path = Path.joinpath(output_filedir,"images")
            images_path = str(images_path) + '/'
            if print_debug_statements: print('images path=',images_path)
            #if not images_path.exists(): Path.mkdir(images_path,parents=True)
            #output_filedir = Path.joinpath(input_filedir, "imaging")
            #if not output_filedir.exists(): Path.mkdir(output_filedir)
        else:
            #input_filepath = main_drive_base_path + 'sort_data_files\\' + input_filename
            images_path = run_data_folder_path_sync + r'imaging\\' #images\\'
        if not os.path.exists(images_path): os.makedirs(images_path)


        source_coordinates = [0, 0, 0]  # [-2.8,0,0] # cm
        NOVO_face_x_coord = 100  # cm

        if "mNOVOv4_PTB" in sim_base_folder_name:
            # extract source position
            source_coords_str = sim_base_folder_name_only.split('_')[1][1:-1].split(',')
            source_coordinates = [float(i) for i in source_coords_str]
            NOVO_face_x_coord = 0  # cm
        elif use_PT_on_tissue_mNOVOv4_settings:
            source_coordinates = [0, 0, -45]
            NOVO_face_x_coord = 0  # cm

        try:
            run_number = int(input_filename[-11:-5])
        except:
            run_number = 99999


    else:  # if using_experimental_data==True
        if on_Linux_machine:
            #main_drive_base_path += 'imaging/mNOVOv3exp_nELBE/'
            #data_drive_base_path += 'imaging/mNOVOv3exp_nELBE/'
            main_drive_base_path += 'imaging/mNOVOv4exp_PTB/'
            data_drive_base_path += 'imaging/mNOVOv4exp_PTB/'
        else:
            #main_drive_base_path += 'imaging\HZDRexp_DTinDE\\'
            #data_drive_base_path += 'imaging\HZDRexp_DTinDE\\'
            data_drive_base_path += 'imaging\mNOVOv3exp_nELBE\\'
            main_drive_base_path += 'imaging\mNOVOv3exp_nELBE\\'
            #main_drive_base_path += 'imaging\mNOVOv4exp_PTB\\'
            #data_drive_base_path += 'imaging\mNOVOv4exp_PTB\\'

        # input_filename = 'sorted_cf252_imaging_000001.root' # Cf252, 100 cm away, Z = 0 cm, low rate
        # input_filename = 'sorted_cf252_imaging_000003.root' # Cf252, 100 cm away, Z = 10 cm, low rate
        # input_filename = 'sorted_cf252_imaging_000004.root' # Cf252, 100 cm away, Z = 7.5 cm, low rate

        # input_filename = 'sorted_cf252_imaging_000007.root' # Cf252, 100 cm away, Z = 0 cm
        # input_filename = 'sorted_cf252_imaging_000008.root' # Cf252, 100 cm away, Z = 0 cm
        # input_filename = 'sorted_cf252_imaging_sum.root' # Cf252, 100 cm away, Z = 0 cm, high stats

        # input_filename = 'sorted_cf252_imaging_000010.root' # Cf252, 100 cm away, Z = 10 cm, high stats

        # input_filename = 'output_pos0.root' # Cf252, 100 cm away, Z = 0 cm, high stats
        # input_filename = 'output_pos1.root' # Cf252, 100 cm away, Z = 10 cm, high stats

        # input_filename = 'cutted_sorted_DT_imaging_000003.root' # DT, 170 cm away, Z = 0 cm, very first DT image
        # input_filename = 'sorted_cutted_DT_imaging_000004.root' # DT, 170 cm away, Z = 0 cm, second first DT image

        # input_filename = 'cutted_sorted2_DT_imaging_000006.root' # DT, 182 cm away, Z = 0 cm, beam current tests
        # input_filename = 'cutted_sorted2_DT_imaging_000007.root' # DT, 182 cm away, Z = 0 cm, beam current tests
        # input_filename = 'cutted_sorted2_DT_imaging_000008.root' # DT, 182 cm away, Z = 0 cm, beam current tests
        # input_filename = 'cutted_sorted2_DT_imaging_000009.root' # DT, 182 cm away, Z = 0 cm, beam current tests
        # input_filename = 'cutted_sorted2_DT_imaging_000010.root' # DT, 182 cm away, Z = 0 cm, beam current tests
        # input_filename = 'cutted_sorted2_DT_imaging_000011.root' # DT, 182 cm away, Z = 0 cm, beam current tests

        # input_filename = 'cutted_sorted2_DT_imaging_000013.root' # DT, 182 cm away, Z = 0 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000015.root' # DT, 182 cm away, Z = -1 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000016.root' # DT, 182 cm away, Z = -2 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000017.root' # DT, 182 cm away, Z = -3 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000018.root' # DT, 182 cm away, Z = -4 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000019.root' # DT, 182 cm away, Z = -5 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000020.root' # DT, 182 cm away, Z = -6 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000021.root' # DT, 182 cm away, Z = -7 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000022.root' # DT, 182 cm away, Z = +1 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000023.root' # DT, 182 cm away, Z = +2 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000024.root' # DT, 182 cm away, Z = +3 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000025.root' # DT, 182 cm away, Z = +4 cm
        # input_filename = 'cutted_sorted2_DT_imaging_000026.root' # DT, 182 cm away, Z = +5 cm

        input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000013.root'  # DT, 182 cm away, Z = 0 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000015.root' # DT, 182 cm away, Z = -1 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000016.root' # DT, 182 cm away, Z = -2 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000017.root' # DT, 182 cm away, Z = -3 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000018.root' # DT, 182 cm away, Z = -4 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000019.root' # DT, 182 cm away, Z = -5 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000020.root' # DT, 182 cm away, Z = -6 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000021.root' # DT, 182 cm away, Z = -7 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000022.root' # DT, 182 cm away, Z = +1 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000023.root' # DT, 182 cm away, Z = +2 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000024.root' # DT, 182 cm away, Z = +3 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000025.root' # DT, 182 cm away, Z = +4 cm
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000026.root' # DT, 182 cm away, Z = +5 cm

        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000028.root' # DT, 182 cm away, Z = 0 cm, high stats
        # input_filename = 'cutted_sorted_DT_imaging_nontimesorted_000029.root' # DT, 182 cm away, Z = +5 cm, high stats

        # nELBE Dec 2023 runs
        input_filename = 'sorted_coinc_2hit_cf252_000109.root'  # Some Cf252 test run with Joey's updated format?



        # now using argparse at top of code
        #if len(sys.argv) > 1:  # allow for specification of input_filename via command line
        #    input_filename = sys.argv[1]
        if use_CLI:
            input_filename = input_path.name

        try:
            run_number = int(input_filename[-11:-5])
        except:
            run_number = 99999

        source_coordinates = [0, 0, 0]  # [-2.8,0,0] # cm
        NOVO_face_x_coord = 182  # cm

        # reject_theta_diff_thresh = -0.05  # negative = fractional diff, positive = absolute diff in radians, None = don't reject any based on angle
        # reject_theta_diff_thresh = 5*(np.pi/180)
        reject_theta_diff_thresh = 90 * (np.pi / 180)
        # reject_theta_diff_thresh = np.pi/2

        bar1_IDs, bar2_IDs = [], []  # all bars allowed
        # bar1_IDs, bar2_IDs = [0,2,4], [0,2,4] # only OGS in both interactions
        # bar1_IDs, bar2_IDs = [1,3,5], [1,3,5] # only M600 in both interactions
        # bar1_IDs, bar2_IDs = [0,2,4], [] # only OGS in first interactions
        # bar1_IDs, bar2_IDs = [1,3,5], [] # only M600 in first interactions
        # bar1_IDs, bar2_IDs = [1,3,5], [0,2,4] # only M600 in first interaction, OGS in second
        # bar1_IDs, bar2_IDs = [3], [4] # only M600 in first interaction, OGS in second

        bar3_IDs = []

        img_tmp_folder_name = ''
        # img_tmp_folder_name = 'OGS_only\\'
        # img_tmp_folder_name = 'M600_only\\'
        # img_tmp_folder_name = 'OGS_first\\'
        # img_tmp_folder_name = 'M600_first\\'
        # img_tmp_folder_name = 'OGS_only_high-res\\'
        # img_tmp_folder_name = 'M600_first_then_OGS\\bar3_to_bar4\\'
        # img_tmp_folder_name = 'test\\'

        #img_tmp_folder_name = 'list_mode_files\\'

        #img_tmp_folder_name = 'test_parallel\\'

        img_tmp_folder_name = input_filename.replace('.root', '')
        img_tmp_folder_name = img_tmp_folder_name.replace('_fastmode', '')
        img_tmp_folder_name = img_tmp_folder_name.replace('.pickle', '')

        if use_fastmode:
            img_tmp_folder_name += '/fastmode'

        # use_true_source_position = True
        use_true_source_position = False

        force_recreate_event_pickle = True  # False # if True, .root file will be forcibly reprocessed, else only reprocessed if pickle absent
        force_recreate_event_pickle = False

        if use_true_source_position:
            source_coordinates[2] = 0  # default to placing source at Z=0, code for exceptions below
            if run_number in [15]:
                source_coordinates[2] = -1
            elif run_number in [16]:
                source_coordinates[2] = -2
            elif run_number in [17]:
                source_coordinates[2] = -3
            elif run_number in [18]:
                source_coordinates[2] = -4
            elif run_number in [19]:
                source_coordinates[2] = -5
            elif run_number in [20]:
                source_coordinates[2] = -6
            elif run_number in [21]:
                source_coordinates[2] = -7
            elif run_number in [22]:
                source_coordinates[2] = 1
            elif run_number in [23]:
                source_coordinates[2] = 2
            elif run_number in [24]:
                source_coordinates[2] = 3
            elif run_number in [25]:
                source_coordinates[2] = 4
            elif run_number in [26]:
                source_coordinates[2] = 5



        if use_CLI:
            input_filepath = str(input_path)
            #output_filepath = output_filedir
            images_path = Path.joinpath(output_filedir,img_tmp_folder_name)
            images_path = str(images_path) + '/'
            if print_debug_statements: print('images path=',images_path)
            #if not images_path.exists(): Path.mkdir(images_path,parents=True)
            #output_filedir = Path.joinpath(input_filedir, "imaging")
            #if not output_filedir.exists(): Path.mkdir(output_filedir)
        else:
            input_filepath = main_drive_base_path + 'sort_data_files\\' + input_filename
            images_path = main_drive_base_path + r'images\\' + img_tmp_folder_name + r'\\'
            image_settings_file = main_drive_base_path + 'sort_data_files\\' + '/image_settings.txt'
            image_settings_filename = 'image_settings.txt'
        if not os.path.exists(images_path): os.makedirs(images_path)

        inputfile_head, inputfile_tail = os.path.split(input_filepath)
        if not use_CLI: output_filedir = inputfile_head + '\\'

        if input_filepath[-5:] == '.root':  # need to find or create pickle
            if use_fastmode:
                image_data_pickle_name = Path.joinpath(output_filedir,os.path.basename(input_filepath).replace('.root', '_fastmode.pickle'))
                #if use_CLI:
                #    image_data_pickle_name = Path.joinpath(output_filedir,os.path.basename(input_filepath).replace('.root', '_fastmode.pickle'))
                #else:
                #    image_data_pickle_name = input_filepath.replace('.root', '_fastmode.pickle')
            else:
                #image_data_pickle_name = input_filepath.replace('.root', '.pickle')
                image_data_pickle_name = Path.joinpath(output_filedir,os.path.basename(input_filepath).replace('.root', '.pickle'))
            print('Imaging data from file: ', image_data_pickle_name)
            # if not os.path.isfile(image_data_pickle_name): # need to convert ROOT file to pickle
            if force_recreate_event_pickle or not os.path.isfile(
                    image_data_pickle_name):  # need to convert ROOT file to pickle
                print('Pickle file does not yet exist, converting it from ROOT file...    ({:0.2f} seconds elapsed)'.format(time.time() - start))
                NOVO_root2numpy(input_filepath, syntax_style='Joey', bar1_IDs=bar1_IDs, bar2_IDs=bar2_IDs, bar3_IDs=bar3_IDs)#, max_n_entries_to_process=max_n_entries_to_process)
                print('Pickle file created!    ({:0.2f} seconds elapsed)'.format(time.time() - start)),
            else:
                print('Pickle file found!    ({:0.2f} seconds elapsed)'.format(time.time() - start))
        elif input_filepath[-7:] == '.pickle':
            image_data_pickle_name = input_filepath
            print('Imaging data from file: ', image_data_pickle_name)


        limit_of_imaging_set_X_events_per_par = None
        # limit_of_imaging_set_X_events_per_par = 1000
        if limit_of_imaging_set_X_events_per_par != None:
            image_data_pickle_name = image_data_pickle_name.replace('.pickle', '_limited_to_{:g}_events.pickle'.format(int(limit_of_imaging_set_X_events_per_par)))


        image_data_pickle_path = image_data_pickle_name

    if use_CLI:
        image_results_pickle_path = Path(str(image_data_pickle_path).replace('.pickle', '_results.pickle'))
    else:
        image_results_pickle_path = image_data_pickle_path.replace('.pickle', '_results.pickle')
    image_results_pickle_path2 = images_path + 'imaging_results.pickle'
    list_mode_image_results_pickle_path = images_path + 'imaging_results_list_mode.pickle'
    results_dict = Munch({})

    save_list_mode_results = True  # saves for each imaged event, event data, cone, and image
    # if use_fastmode: save_list_mode_results = False
    # save_list_mode_results = False
    temporary_disable_list_mode_pickle_save = False  # True # Enablig this lets the code run normally with save_list_mode_results = True BUT does not write the big pickle file

    use_MC_Truth_xyz_coordinates = True  # if true, uses "neutron_records", if false uses "neutron_records_exp" (a format exclusive to MC results)
    #use_MC_Truth_xyz_coordinates = False

    MC_truth_extra_data_available = True
    if use_MC_Truth_xyz_coordinates:
        MCtrutxt = 'MCtrue_'
    else:
        MCtrutxt = 'MCexp_'
    if using_experimental_data:
        MCtrutxt = 'exp_'
        MC_truth_extra_data_available = False

    if not using_experimental_data:
        image_results_pickle_path2 = image_results_pickle_path2.replace('.pickle','_'+MCtrutxt[:-1]+'.pickle')
        list_mode_image_results_pickle_path = list_mode_image_results_pickle_path.replace('.pickle','_'+MCtrutxt[:-1]+'.pickle')

    max_num_cones = None  # set to None to not limit number of cones, otherwise set to int limit of cones

    # images_path = main_drive_base_path + r'imaging\\images\\'
    #if not using_experimental_data:
    #    images_path = imaging_code_testing_path + r'images\\' + sim_base_folder_name_only + '\\'
    #else:
    #    images_path = main_drive_base_path + r'images\\' + img_tmp_folder_name + r'\\'
    #if not os.path.exists(images_path): os.makedirs(images_path)

    '''
    #im_half_dim = .2 # cm
    #pixel_vwidth = 0.002 # cm
    im_half_dim = 5 #.2 # cm
    pixel_vwidth = 0.1 #0.002 # cm
    pixel_hwidth = pixel_vwidth # cm
    source_coordinates = [0,0,0] #[-2.8,0,0] # cm
    max_En0MeV_allowed = 25 #14.2 # MeV
    NOVO_face_x_coord = 100 # cm
    if 'd25cm' in sim_base_folder_name:
        NOVO_face_x_coord = 25 # cm

    '''



    # im_half_dim = .2 # cm
    # pixel_vwidth = 0.002 # cm
    im_half_dim = 50  # .2 # cm
    pixel_vwidth = 1  # 0.002 # cm
    pixel_hwidth = pixel_vwidth  # cm
    max_En0MeV_allowed = 150 #25  # 14.2 # MeV
    if not using_experimental_data:
        if 'd25cm' in sim_base_folder_name:
            NOVO_face_x_coord = 25  # cm

    # im_half_dim = 100 #.2 # cm
    # pixel_vwidth = 1 #0.002 # cm
    # pixel_hwidth = 0.2 # cm

    im_half_dim = 100  # cm
    pixel_vwidth = 0.5  # 0.2 # cm
    # pixel_vwidth = 1 #0.002 # cm
    pixel_hwidth = pixel_vwidth  # cm


    use_parallelization_for_imaging = True
    if use_CLI:
        num_processes = num_processes
        if num_processes==0:
            use_parallelization_for_imaging = False
            print('Not using parallelization for imaging...')
        elif num_processes==None:
            print('Using all available processes for parallelization for imaging...')
        else:
            print('Using {:g} processes for parallelization for imaging...'.format(num_processes))
    else:
        num_processes = 3 # None results in using max available processes

    make_neutron_cones = True
    make_gamma_cones = True
    image_neutron_events = make_neutron_cones
    image_gamma_events = make_gamma_cones

    image_via_matrix_math = True
    image_via_ray_tracing = not image_via_matrix_math
    iteratively_search_for_cone_image_bounds = False  # True # should make smoother curves on image but increases CPU time


    axis_normal_to_array_face = 'z'

    # fastmode imaging settings
    fast_max_num_imaged_cones = 2000
    fast_im_half_dim = 100  # cm
    fast_pixel_vwidth = 2  # cm
    fast_pixel_hwidth = fast_pixel_vwidth  # cm
    fast_make_neutron_cones = True
    fast_make_gamma_cones = True
    fast_image_neutron_events = make_neutron_cones
    fast_image_gamma_events = make_gamma_cones

    source_monoEn_MeV = 14.1


    if os.path.isfile(image_settings_file):
        print(f"Overwriting default image settings with those in {image_settings_filename}")
        f = open(image_settings_file)
        ignore_settings_file = False
        for line in f:
            if ignore_settings_file: break
            exec(line)
        f.close()
    '''
    Example contents of an image_settings.txt file:
    FOR PHITS SIMULATIONS:

    # The below lines are executed as code in the expNOVO_imager.py script
    # To disable this file without deleting it, uncomment the "ignore_settings_file = True" line below
    #   Any lines above it will still be executed. Individual lines can be commented out too.
    # global ignore_settings_file          ; ignore_settings_file           = True

    global skip_gamma_events             ; skip_gamma_events              = True  # if True, events with gamma timing are skipped
    global start_det_to_hit1_max_gtime_ns; start_det_to_hit1_max_gtime_ns = 20.0  # if abs(tstart-thit1)<this, assume incident par is gamma
    global start_det_to_hit1_max_ntime_ns; start_det_to_hit1_max_ntime_ns = 300.0 # if abs(tstart-thit1)<this, assume incident par is ultra slow neutron and skip it
    global min_allowed_tof12_ns          ; min_allowed_tof12_ns           = 2.0 # reject events with shorter tof values (only if skip_gamma_events = True)
    global min_allowed_gamma_tof12_ns    ; min_allowed_gamma_tof12_ns     = 0.5 # reject events with shorter tof values (only if skip_gamma_events = False)
    global max_allowed_tof12_ns          ; max_allowed_tof12_ns           = 20.0 # reject events with longer tof values
    global max_allowed_calc_E0           ; max_allowed_calc_E0            = 150.0 # reject events whose neutron ToF energy > this (in MeV)

    # fastmode_filters
    global fast_n_min_elong1; fast_n_min_elong1 = 0.1 # MeVee
    global fast_n_min_elong2; fast_n_min_elong2 = 0.1 # MeVee
    global fast_g_min_elong ; fast_g_min_elong  = 0.5 # MeVee

    # normal mode filters
    global n_min_elong1; n_min_elong1 = 0.3 # MeVee
    global n_min_elong2; n_min_elong2 = 0.3 # MeVee
    global g_min_elong ; g_min_elong  = 0.1 # MeVee

    # experiment settings
    global source_type              ; source_type               = 'DT' # choose between 'Cf252', 'nELBE', and 'DT' ; this controls how En0 is determined
    # global source_coordinates     ; source_coordinates        = [0, 0, -549.3]  # cm, value for nELBE source
    global source_coordinates       ; source_coordinates        = [0, 0, -45]  # cm, value for PHITS simulation
    global NOVO_face_x_coord        ; NOVO_face_x_coord         = 0  # cm; is actually on the axis specified below
    global axis_normal_to_array_face; axis_normal_to_array_face = 'z' # select 'x', 'y', or 'z'

    # standard imaging settings
    im_half_dim = 20  # cm
    pixel_vwidth = 0.1 # cm
    pixel_hwidth = pixel_vwidth  # cm
    max_En0MeV_allowed = 150  # MeV
    make_neutron_cones = True
    make_gamma_cones = True
    image_neutron_events = make_neutron_cones
    image_gamma_events = make_gamma_cones

    # fastmode imaging settings
    global fast_max_num_imaged_cones; fast_max_num_imaged_cones = 20000
    fast_im_half_dim = 20  # cm
    fast_pixel_vwidth = 0.5  # cm
    fast_pixel_hwidth = fast_pixel_vwidth  # cm
    fast_make_neutron_cones = True
    fast_make_gamma_cones = True
    fast_image_neutron_events = make_neutron_cones
    fast_image_gamma_events = make_gamma_cones



    FROM PTB EXPERIMENT FOR 14 MEV NEUTRONS:

    # The below lines are executed as code in the expNOVO_imager.py script
    # To disable this file without deleting it, uncomment the "ignore_settings_file = True" line below
    #   Any lines above it will still be executed. Individual lines can be commented out too.
    # global ignore_settings_file          ; ignore_settings_file           = True

    global skip_gamma_events             ; skip_gamma_events              = False  # if True, events with gamma timing are skipped
    global start_det_to_hit1_max_gtime_ns; start_det_to_hit1_max_gtime_ns = 20.0  # if abs(tstart-thit1)<this, assume incident par is gamma
    global start_det_to_hit1_max_ntime_ns; start_det_to_hit1_max_ntime_ns = 300.0 # if abs(tstart-thit1)<this, assume incident par is ultra slow neutron and skip it
    global min_allowed_tof12_ns          ; min_allowed_tof12_ns           = 0.5 # reject events with shorter tof values (only if skip_gamma_events = True)
    global min_allowed_gamma_tof12_ns    ; min_allowed_gamma_tof12_ns     = 0.05 # reject events with shorter tof values (only if skip_gamma_events = False)
    global max_allowed_tof12_ns          ; max_allowed_tof12_ns           = 25.0 # reject events with longer tof values
    global max_allowed_calc_E0           ; max_allowed_calc_E0            = 25.0 # reject events whose neutron ToF energy > this (in MeV)
    global source_monoEn_MeV             ; source_monoEn_MeV              = 14.8 # reject events whose neutron ToF energy > this (in MeV)
    global max_n_entries_to_process      ; max_n_entries_to_process       = None # limit the number of events converted from ROOT to NumPy, None = no limit

    # fastmode_filters
    global fast_n_min_elong1; fast_n_min_elong1 = 3.0 #0.5 # MeVee
    global fast_n_min_elong2; fast_n_min_elong2 = 1.0 #0.5 # MeVee
    global fast_g_min_elong ; fast_g_min_elong  = 0.5 # MeVee

    # normal mode filters
    global n_min_elong1; n_min_elong1 = 0.3 # MeVee
    global n_min_elong2; n_min_elong2 = 0.3 # MeVee
    global g_min_elong ; g_min_elong  = 0.1 # MeVee

    # experiment settings
    global source_type              ; source_type               = 'DT' # choose between 'Cf252', 'nELBE', and 'DT' ; this controls how En0 is determined
    global source_coordinates       ; source_coordinates        = [0, 0, -109.6]  # cm, value for nELBE source
    # global source_coordinates       ; source_coordinates        = [0, 0, 110.2]  # cm, value for Cf252 source in nELBE cave, placed downstream from array
    global NOVO_face_x_coord        ; NOVO_face_x_coord         = 0  # cm; is actually on the axis specified below
    global axis_normal_to_array_face; axis_normal_to_array_face = 'z' # select 'x', 'y', or 'z'

    # standard imaging settings
    im_half_dim = 100  # cm
    pixel_vwidth = 0.5 # cm
    pixel_hwidth = pixel_vwidth  # cm
    max_En0MeV_allowed = 25  # MeV
    make_neutron_cones = True
    make_gamma_cones = True
    image_neutron_events = make_neutron_cones
    image_gamma_events = make_gamma_cones

    # fastmode imaging settings
    global fast_max_num_imaged_cones; fast_max_num_imaged_cones = 20000
    fast_im_half_dim = 100  # cm
    fast_pixel_vwidth = 2  # cm
    fast_pixel_hwidth = fast_pixel_vwidth  # cm
    fast_make_neutron_cones = True
    fast_make_gamma_cones = True
    fast_image_neutron_events = make_neutron_cones
    fast_image_gamma_events = make_gamma_cones

    '''

    print('Setting source coordinates at: ', source_coordinates, 'for run number', run_number)

    if use_fastmode:
        # fastmode imaging settings
        #fast_max_num_imaged_cones = fast_max_num_imaged_cones
        im_half_dim = fast_im_half_dim
        pixel_vwidth = fast_pixel_vwidth
        pixel_hwidth = fast_pixel_hwidth
        make_neutron_cones   = fast_make_neutron_cones
        make_gamma_cones     = fast_make_gamma_cones
        image_neutron_events = fast_image_neutron_events
        image_gamma_events   = fast_image_gamma_events






    if not using_experimental_data:
        # preset values for particular runs
        if sim_base_folder_name == 'parallel_3d2t_shifted-ring-source':
            reject_theta_diff_thresh = -0.15  # negative = fractional diff, positive = absolute diff, None = don't reject any based on angle
            im_half_dim = 14  # cm
            pixel_vwidth = 0.2  # cm
            pixel_hwidth = pixel_vwidth  # cm
            source_coordinates = [0, 3, 4]  # cm
            max_En0MeV_allowed = 25  # 14.2 # MeV
            NOVO_face_x_coord = 100  # cm

        # manually override with a data file from not my own naming system
        use_Lena_test_data = False
        # use_Lena_test_data = True
        if use_Lena_test_data:
            sim_base_folder_name_only = 'Lena_data'
            image_data_pickle_name = 'true_NCE.pickle'
            run_data_folder_path_sync = imaging_other_data_path + 'test_files\\'
            reject_theta_diff_thresh = -0.05
            im_half_dim = 0.05  # 10  # cm
            pixel_vwidth = 0.0005  # cm
            pixel_hwidth = pixel_vwidth  # cm
            source_coordinates = [20, 0, 0]  # cm
            max_En0MeV_allowed = 1.05  # MeV
            NOVO_face_x_coord = 0  # cm
            reject_theta_diff_thresh = None

        #image_data_pickle_path = run_data_folder_path_sync + image_data_pickle_name

        if not os.path.exists(image_data_pickle_path):
            print('The following imaging data pickle file could not be found, aborting!')
            print(image_data_pickle_path)




    '''
    #images_path = main_drive_base_path + r'imaging\\images\\'
    if not using_experimental_data:
        images_path = imaging_code_testing_path + r'images\\' + sim_base_folder_name_only + '\\'
    else:
        images_path = main_drive_base_path +  r'images\\' + img_tmp_folder_name +  r'\\'
    if not os.path.exists(images_path): os.makedirs(images_path)
    '''

    '''
    Set up imaging plane (necessary info for gamma cone construction)
    '''
    # pick a plane for image. One coordinate MUST match between the two, and the plane should also pass through the source
    # image_plane_bottom_left_corner = [0,-50,-50]
    # image_plane_top_right_corner = [0,50,50]
    im_half_dim = im_half_dim
    if axis_normal_to_array_face.lower()=='x':
        image_plane_bottom_left_corner = [source_coordinates[0], -1 * im_half_dim, -1 * im_half_dim]
        image_plane_top_right_corner = [source_coordinates[0], im_half_dim, im_half_dim]
    elif axis_normal_to_array_face.lower()=='y':
        image_plane_bottom_left_corner = [-1 * im_half_dim, source_coordinates[1], -1 * im_half_dim]
        image_plane_top_right_corner = [im_half_dim, source_coordinates[1], im_half_dim]
    elif axis_normal_to_array_face.lower()=='z':
        image_plane_bottom_left_corner = [ -1 * im_half_dim, -1 * im_half_dim, source_coordinates[2]]
        image_plane_top_right_corner = [im_half_dim, im_half_dim, source_coordinates[2]]
    else:
        print('Invalid argument provided to "axis_normal_to_array_face"; options are "x", "y", and "z"')
    # pixel_vwidth = 1
    # pixel_hwidth = pixel_vwidth

    # check plane
    img_blc = image_plane_bottom_left_corner
    img_trc = image_plane_top_right_corner
    count_matches = 0
    non_match_i = []
    for i in range(3):
        if img_blc[i] == img_trc[i]:
            i_match = i
            val_match = img_blc[i]
            count_matches += 1
        else:
            non_match_i.append(i)
    if count_matches != 1:
        print(
            'Imaging plane incorrect. One and ONLY one coordinate must match between the corners specifying the plane')
        print(img_blc, img_trc)
        sys.exit()

    planes_list = ['YZ', 'ZX', 'XY']
    axes_list = ['X', 'Y', 'Z']

    print('Plane parallel to {:} plane at {:} = {:g} spanning {:}[{:g},{:g}] and {:}[{:g},{:g}] selected.'.format(
        planes_list[i_match], axes_list[i_match], img_blc[i_match],
        axes_list[non_match_i[0]], img_blc[non_match_i[0]], img_trc[non_match_i[0]],
        axes_list[non_match_i[1]], img_blc[non_match_i[1]], img_trc[non_match_i[1]]))

    if planes_list[i_match] == 'YZ':
        non_match_i = non_match_i[::-1]  # in this case, we'll want Z on the horizontal axis

    vhat_blc = np.array(img_blc) / magnitude(img_blc)
    vhat_trc = np.array(img_trc) / magnitude(img_trc)

    pa = np.array(img_blc)
    pb = np.array(img_trc)
    pa_hat = vhat_blc
    pb_hat = vhat_trc
    if i_match == 0:
        pc = [val_match, pa[1], pb[2]]
        im_lr_range = [image_plane_bottom_left_corner[1], image_plane_top_right_corner[1]]
        im_bt_range = [image_plane_bottom_left_corner[2], image_plane_top_right_corner[2]]
    elif i_match == 1:
        pc = [pa[0], val_match, pb[2]]
        im_lr_range = [image_plane_bottom_left_corner[0], image_plane_top_right_corner[0]]
        im_bt_range = [image_plane_bottom_left_corner[2], image_plane_top_right_corner[2]]
    else:
        pc = [pa[0], pb[1], val_match]
        im_lr_range = [image_plane_bottom_left_corner[0], image_plane_top_right_corner[0]]
        im_bt_range = [image_plane_bottom_left_corner[1], image_plane_top_right_corner[1]]
    pc = np.array(pc)
    pc_hat = pc / magnitude(pc)

    # print(pa,pb,pc)

    hbin_edges = np.arange(im_lr_range[0], im_lr_range[1] + pixel_hwidth, pixel_hwidth)
    vbin_edges = np.arange(im_bt_range[0], im_bt_range[1] + pixel_vwidth, pixel_vwidth)
    num_hbins = len(hbin_edges) - 1
    num_vbins = len(vbin_edges) - 1
    hbin_mids = 0.5 * (hbin_edges[1:] + hbin_edges[:-1])
    vbin_mids = 0.5 * (vbin_edges[1:] + vbin_edges[:-1])

    # num_hbins = int((im_lr_range[1] - im_lr_range[0])/pixel_hwidth)
    # num_vbins = int((im_bt_range[1] - im_bt_range[0])/pixel_vwidth)

    plane_normal = np.cross(pa_hat - pb_hat, pb_hat - pc_hat)
    plane_normal = plane_normal / magnitude(plane_normal)
    # print(plane_normal)
    # sys.exit()

    results_dict.meta = Munch({
        'source_coordinates': source_coordinates,
        'NOVO_face_x_coord': NOVO_face_x_coord,
        'reject_theta_diff_thresh': reject_theta_diff_thresh,
        'img_blc': img_blc,
        'img_trc': img_trc,
        'i_match': i_match,
        'non_match_i': non_match_i,
        'im_lr_range': im_lr_range,
        'im_bt_range': im_bt_range,
        'im_half_dim': im_half_dim,
        'pixel_vwidth': pixel_vwidth,
        'pixel_hwidth': pixel_hwidth,
        'max_En0MeV_allowed': max_En0MeV_allowed,
        'hbin_edges': hbin_edges,
        'num_hbins': num_hbins,
        'hbin_mids': hbin_mids,
        'vbin_edges': vbin_edges,
        'num_vbins': num_vbins,
        'vbin_mids': vbin_mids,
        'plane_normal': plane_normal,
        'image_data_pickle_path': image_data_pickle_path
    })

    figi = 10

'''
██████  ██    ██ ██ ██      ██████       ██████  ██████  ███    ██ ███████ ███████ 
██   ██ ██    ██ ██ ██      ██   ██     ██      ██    ██ ████   ██ ██      ██      
██████  ██    ██ ██ ██      ██   ██     ██      ██    ██ ██ ██  ██ █████   ███████ 
██   ██ ██    ██ ██ ██      ██   ██     ██      ██    ██ ██  ██ ██ ██           ██ 
██████   ██████  ██ ███████ ██████       ██████  ██████  ██   ████ ███████ ███████ 
'''

if __name__ == '__main__':
    print('\t Loading {} ...  ({:0.2f} seconds elapsed)'.format(image_data_pickle_path, time.time() - start))
    with open(image_data_pickle_path, 'rb') as handle:
        imaging_data_dict = pickle.load(handle)
    print('\t Loaded  {}  !   ({:0.2f} seconds elapsed)\n'.format(image_data_pickle_path, time.time() - start))

    if use_MC_Truth_xyz_coordinates or using_experimental_data:
        neutron_records = imaging_data_dict['neutron_records']
        gamma_records = imaging_data_dict['gamma_records']
    else:
        # "Exp" versions use bar center coordinates + MC exact DOI (instead of all MC exact true values)
        if MC_truth_extra_data_available:
            neutron_records = imaging_data_dict['neutron_records_exp']
            gamma_records = imaging_data_dict['gamma_records_exp']
        else:
            print('MC_truth_extra_data_available==False, therefore smeared MC data will NOT be used')
            neutron_records = imaging_data_dict['neutron_records']
            gamma_records = imaging_data_dict['gamma_records']

    num_neutron_records = len(neutron_records)
    num_gamma_records = len(gamma_records)

    print('{:g} neutron records and {:g} gamma records found   ({:0.2f} seconds elapsed)'.format(num_neutron_records,
                                                                                                 num_gamma_records,
                                                                                                 time.time() - start))
    if num_neutron_records == 0:
        print('No neutron records found, not building cones for or imaging neutron events...')
        make_neutron_cones = False
        image_neutron_events = make_neutron_cones
    if num_gamma_records == 0:
        print('No gamma records found, not building cones for or imaging gamma events...')
        make_gamma_cones = False
        image_gamma_events = make_gamma_cones

    event_cone_type = np.dtype([('x', np.single), ('y', np.single), ('z', np.single),  # cone origin/apex coordinates
                                ('u', np.single), ('v', np.single), ('w', np.single),  # cone direction unit vector
                                ('theta', np.single), ('theta_MCtruth', np.single)])  # cone angle

    n_cones = np.empty((num_neutron_records), dtype=event_cone_type)  # neutron imaging records
    i_nc = 0  # current index of neutron cones
    indices_of_kept_event_records_n = []
    g_cones = np.empty((6 * num_gamma_records), dtype=event_cone_type)  # neutron imaging records
    i_gc = 0  # current index of neutron cones
    indices_of_kept_event_records_g = []

    good_proton_count, good_carbon_count, proton_count, carbon_count = 0, 0, 0, 0

    # build_neutron_cone_function
    def build_neutron_cone(e, max_En0MeV_allowed=None, guess_source_loc=[0, 0, 0], reject_theta_diff_thresh=None):
        global using_experimental_data
        '''
        Input: a neutron event 'e', a numpy array entry with the following dtype:
        neutron_event_record_type = np.dtype([('type', 'S1'),
                                     ('x1', np.single),('y1', np.single),('z1', np.single),('t1', np.single),('dE1', np.single),
                                     ('x2', np.single),('y2', np.single),('z2', np.single),('t2', np.single),
                                     ('protons_only',np.bool_),('theta_MCtruth', np.single)])

        Output: a neutron event cone with the following dtype:
        event_cone_type = np.dtype([('x', np.single),('y', np.single),('z', np.single), # cone origin/apex coordinates
                                ('u', np.single),('v', np.single),('w', np.single), # cone direction unit vector
                                ('theta',np.single),('theta_MCtruth', np.single)]) # cone angle

        Method:
        A neutron event with scatters in two bars, with the following information
           - 1st scatter : (x1,y1,z1), dE1 (proton energy deposited), t1 (timestamp 1)
           - 2nd scatter : (x2,y2,z2), t2 (timestamp 2)
        The positions and times of the two scatters allows a time-of-flight calculation to determine En', the neutron
        energy after the first scatter. The cone angle is then theta = arcsin(sqrt(Ep/En')). This cone has
        its axis on the line connecting the two scatters with its apex at the coordinates of the first scatter.
        '''
        global proton_count, carbon_count, good_proton_count, good_carbon_count
        if max_En0MeV_allowed != None:
            En_allowed_max_MeV = max_En0MeV_allowed
        else:
            En_allowed_max_MeV = 10000  # 10 GeV limit
        # En_allowed_max_MeV = 14.2 # hard coded maximum inbound neutron energy, will reject events exceeding this
        tof = e['t2'] - e['t1']  # ns
        d = dist([e['x1'], e['y1'], e['z1']], [e['x2'], e['y2'], e['z2']])  # cm
        v = d / tof  # cm/ns

        if d > 50:
            print('Bad neutron event provided')
            print('(x1,y1,z1)=', [e['x1'], e['y1'], e['z1']], '(x2,y2,z2)=', [e['x2'], e['y2'], e['z2']])
            # print('t1=',e['t1'],'  t2=',e['t2'])
            print('dist=', d)  # ,'ToF=',tof,'velocity=',v)
            print()
            return False
        if tof < 0:
            print('Bad neutron event provided')
            # print('(x1,y1,z1)=',[e['x1'],e['y1'],e['z1']],'(x2,y2,z2)=',[e['x2'],e['y2'],e['z2']])
            print('t1=', e['t1'], '  t2=', e['t2'])
            print('ToF=', tof)  # 'dist=',d,'velocity=',v)
            print()
            return False

        mn = 939.565  # MeV/c^2
        mp = 938.272  # MeV/c^2
        mC = (12 * 931.494) - (6 * 0.511)  # MeV/c^2

        L_gamma = Lorentz_gamma(v)  # unitless
        En_prime = (L_gamma - 1) * mn  # MeV
        E_deposited = e['dE1']  # MeV

        En0 = En_prime + E_deposited
        # print(En0)

        if En0 > En_allowed_max_MeV:
            print('Bad neutron event provided')
            print('(x1,y1,z1)=', [e['x1'], e['y1'], e['z1']], '(x2,y2,z2)=', [e['x2'], e['y2'], e['z2']])
            print('t1=', e['t1'], '  t2=', e['t2'])
            print('dist=', d, 'ToF=', tof, 'velocity=', v)
            print('En_prime=', En_prime, 'E_deposited=', E_deposited, 'E_inbound=', E_deposited + En_prime)
            print()
            return False

        dir_vect = unit_vector(np.array([e['x1'] - e['x2'], e['y1'] - e['y2'], e['z1'] - e['z2']]))
        gsl = np.array(guess_source_loc)
        guess_source_vec = unit_vector(np.array([gsl[0] - e['x1'], gsl[1] - e['y1'], gsl[2] - e['z1']]))
        ang_between_dir_and_guess = angle_between(dir_vect, guess_source_vec)

        # https://farside.ph.utexas.edu/teaching/336k/Newton/node52.html
        # https://dspace.mit.edu/bitstream/handle/1721.1/74136/22-05-fall-2006/contents/lecture-notes/lecture14.pdf

        # First assume proton elastic scattering
        m_recoil = mp  # assume proton recoil
        A = m_recoil / mn

        arg1 = 1 - ((E_deposited / En0) * ((1 + A) ** 2) / (2 * A))
        theta_CoM = np.arccos(arg1)
        # theta = np.arctan(np.sin(theta_CoM) / (np.cos(theta_CoM) + (1 / A)))
        # if theta < 0:
        #   theta += np.pi
        theta = np.arctan2(np.sin(theta_CoM), (np.cos(theta_CoM) + (1 / A)))
        # arg2b = (1+A*np.cos(theta_CoM)) / np.sqrt( A**2 + 2*A*np.cos(theta_CoM) + 1 )
        # theta = np.arccos( arg2b )

        theta_proton = theta

        # Now assume carbon elastic scattering
        m_recoil = mC  # assume proton recoil
        A = m_recoil / mn

        arg1 = 1 - ((E_deposited / En0) * ((1 + A) ** 2) / (2 * A))
        theta_CoM = np.arccos(arg1)
        # theta = np.arctan(np.sin(theta_CoM) / (np.cos(theta_CoM) + (1 / A)))
        # if theta < 0:
        #    theta += np.pi
        theta = np.arctan2(np.sin(theta_CoM), (np.cos(theta_CoM) + (1 / A)))
        # arg2b = (1 + A * np.cos(theta_CoM)) / np.sqrt(A ** 2 + 2 * A * np.cos(theta_CoM) + 1)
        # theta = np.arccos(arg2b)

        theta_carbon = theta
        # for many proton events, abs(arg)>1, therefore theta_CoM and then theta(_carbon) are NaNs

        # print(theta,max_scatter_ang_w_proton)

        # Compare the two angles against the guessed angle between apex-to-source and dir vector
        if abs(ang_between_dir_and_guess - theta_proton) < abs(ang_between_dir_and_guess - theta_carbon) or np.isnan(
                theta_carbon):
            theta = theta_proton
            proton_count += 1
        else:
            theta = theta_carbon
            carbon_count += 1

        if reject_theta_diff_thresh == None:
            pass
        elif reject_theta_diff_thresh >= 0:  # absolute difference threshold, in radians
            diff = abs(theta - ang_between_dir_and_guess)
            if diff > reject_theta_diff_thresh:
                return False
        else:
            diff = abs(theta - ang_between_dir_and_guess) / ang_between_dir_and_guess
            if diff > abs(reject_theta_diff_thresh):
                return False

        if theta == theta_proton:
            good_proton_count += 1
        else:
            good_carbon_count += 1

        # uncomment to override use of source location to decide which particle it likely is
        # theta = theta_proton
        # theta = theta_carbon

        # print(theta)

        # theta = np.arccos( np.sqrt( (E_deposited/En0)*((1+A)**2)/(4*A) ) )
        # This math works for proton elastic scattering
        # theta = np.arcsin(np.sqrt((E_deposited / En0) * ((1 + A) ** 2) / (4 * A)))

        # This math almost works for Carbon elastic scatters, to reproduce MC truth angles
        # theta = 2*np.arcsin( np.sqrt( (E_deposited/En0)*((1+A)**2)/(4*A) ) )

        # dir_vect = unit_vector(np.array([e['x1']-e['x2'],e['y1']-e['y2'],e['z1']-e['z2']]))
        # print('(x1,y1,z1)=', [e['x1'], e['y1'], e['z1']], '(x2,y2,z2)=', [e['x2'], e['y2'], e['z2']])
        # print(theta*180/np.pi,dir_vect,En0)
        # print()

        cone = np.empty((1), dtype=event_cone_type)
        cone['x'][0] = e['x1']
        cone['y'][0] = e['y1']
        cone['z'][0] = e['z1']
        cone['u'][0] = dir_vect[0]
        cone['v'][0] = dir_vect[1]
        cone['w'][0] = dir_vect[2]
        cone['theta'][0] = theta
        if using_experimental_data:
            cone['theta_MCtruth'][0] = ang_between_dir_and_guess
        else:
            cone['theta_MCtruth'][0] = e['theta_MCtruth']

        # print(theta/e['theta_MCtruth'])
        return cone


    if make_neutron_cones:
        protons_only_counter = 0
        bad_events_supplied = 0
        i_gnc = 0

        print("\nBuilding neutron cones...   ({:0.2f} seconds elapsed)".format(time.time() - start))
        for i in range(num_neutron_records):
            this_event_record = neutron_records[i_nc]
            if this_event_record['protons_only']: protons_only_counter += 1

            this_cone = build_neutron_cone(this_event_record, max_En0MeV_allowed=max_En0MeV_allowed,
                                           guess_source_loc=source_coordinates,
                                           reject_theta_diff_thresh=reject_theta_diff_thresh)
            if this_cone:
                n_cones[i_gnc] = this_cone
                i_gnc += 1
                indices_of_kept_event_records_n.append(i_nc)
                if use_fastmode and i_gnc>fast_max_num_imaged_cones: break
            else:
                bad_events_supplied += 1

            i_nc += 1

        num_good_n_cones = i_gnc

        if num_good_n_cones==0:
            image_neutron_events = False

        n_cones = n_cones[:i_gnc]

        print('Identified {:g} proton cones and {:g} carbon cones'.format(proton_count, carbon_count))
        print(protons_only_counter, 'proton only events')
        print(bad_events_supplied, 'bad events supplied')
        print(num_good_n_cones, 'total good cones to image')
        print('Consisting of {:g} good proton cones and {:g} good carbon cones'.format(good_proton_count,
                                                                                       good_carbon_count))

        theta_calc_list = []
        theta_MCtruth_list = []
        for ic in range(len(n_cones)):
            theta_calc_list.append(n_cones['theta'][ic])
            theta_MCtruth_list.append(n_cones['theta_MCtruth'][ic])

        num_cones = len(theta_calc_list)

        title_str = r'$\theta$ calculated vs MC truth'
        fig1theta = plt.figure(figi)
        ax1theta = fig1theta.add_subplot(1, 1, 1, rasterized=True)
        fig1theta, ax1theta = fancy_plot(
            [theta_calc_list],
            [theta_MCtruth_list],
            x_scale='linear',
            y_scale='linear',
            x_label_str=r'$\theta$ calculated [radians]',
            y_label_str=r'$\theta$ MC truth [radians]',
            title_str=title_str,
            # marker='.',
            # linestyle='-',
            color='k',
            alpha=0.2,
            x_limits=[0, 1.1 * np.pi],
            y_limits=[0, 1.1 * np.pi],
            fig_width_inch=6.5,
            fig_height_inch=4.5,
            figi=figi,
            fig=fig1theta,
            ax=ax1theta
        )
        figi += 1
        # plt.xticks(np.arange(0, np.pi+np.pi/4, step=(np.pi/4)), ['0','π/4','π/2','3π/4','π'])
        # plt.yticks(np.arange(0, np.pi+np.pi/4, step=(np.pi/4)), ['0','π/4','π/2','3π/4','π'])
        axes = plt.gca()
        slx, sly = 0.5, 0.98
        text_on_plot = '{:g} neutron event cones'.format(num_cones)
        axes.text(slx, sly, text_on_plot, color='black', horizontalalignment='center', verticalalignment='top',
                  transform=axes.transAxes, fontsize=16)  # , weight='bold')
        if save_plots:
            for ext in image_extensions:
                plot_save_path = images_path + 'neutron_' + MCtrutxt + slugify(
                    title_str) + ext  # or use fig.canvas.get_window_title()
                fig1theta.savefig(plot_save_path, facecolor=(0, 0, 0, 0), dpi=500, bbox_inches='tight')

        # plt.show()
        # sys.exit()

    # build_gamma_cone_function
    def build_gamma_cone(e, image_plane_norm_vec=None, max_En0MeV_allowed=None, guess_source_loc=[0, 0, 0],
                         return_only_best_cone=True, reject_theta_diff_thresh=None):
        '''
        Input: a gamma event 'e', a numpy array entry with the following dtype:
        neutron_event_record_type = np.dtype([('type', 'S1'),
                                     ('x1', np.single),('y1', np.single),('z1', np.single),('dE1', np.single),
                                     ('x2', np.single),('y2', np.single),('z2', np.single),('dE2', np.single),
                                     ('x3', np.single),('y3', np.single),('z3', np.single),('dE3', np.single),
                                     ('t1', np.single),('t2', np.single),('t3', np.single),('theta1_MCtruth', np.single)])
        image_plane_norm_vec = a unit vector of the normal of the imaging plane, pointing AWAY from the NOVO face

        Output: a neutron event cone with the following dtype:
        event_cone_type = np.dtype([('x', np.single),('y', np.single),('z', np.single), # cone origin/apex coordinates
                                ('u', np.single),('v', np.single),('w', np.single), # cone direction unit vector
                                ('theta',np.single),('theta_MCtruth', np.single)]) # cone angle

        Method:
        A gamma event with scatters in three bars, with the following information
           - 1st scatter : (x1,y1,z1), dE1
           - 2nd scatter : (x2,y2,z2), dE2
           - 3rd scatter : (x3,y3,z3), dE3
        First, the second scattering angle theta2 is calculated from the three positions. Then that is used with dE1 and dE2
        to find the incident gamma energy Eg via Compton kinematics.  Then, the first scattering angle can be calculated
        solely from that Eg and Eg-dE1.  This first scattering angle is the cone angle theta. This cone has
        its axis on the line connecting the two scatters with its apex at the coordinates of the first scatter.
        The challenge is that, in the experimental data, there is no way of knowing the true order of these scatters.
        Thus, the various posibilities must be tested and the most probable one identified.
        '''
        if isinstance(image_plane_norm_vec, type(None)):
            print(
                "'image_plane_norm_vec' must be provided for determining which gamma cones point toward imaging plane, quitting...")
            sys.exit()
        me = 0.511  # MeV/c^2

        theta1_min = (1.0) * np.pi / 180  # angles lower than this are rejected as they'd not even be detected

        arrs = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1],
                [2, 1, 0]]  # possible arrangements of the 3 scatters
        elim_arrs = []  # arrangements eliminated
        keep_arrs = []  # arrangements kept
        kept_theta1s = []  # list of theta1 values for the arrs values not culled here
        kept_ang_between_dir_and_guess = []
        kept_dir_vecs = []
        kept_s1_locs = []
        kept_thetacalc_thetaguess_absdiffs = []
        for i_arr, arr in enumerate(arrs):
            i = arr[0]  # first scatter
            j = arr[1]  # second scatter
            k = arr[2]  # third scatter

            # First, determine which arrangements first two scatters yield cones pointing toward the imaging plane (if pointing away, eliminate)
            # On second though, first scatter can be a backscatter?
            # if np.dot(dir_vec,image_plane_norm_vec) < 0:
            #    # cone points away from imaging plane, reject it
            #    elim_arrs.append(arr)
            #    continue
            # keep_arrs.append(arr)
            # kept_dir_vecs.append(dir_vec)

            # Calculate theta2
            vec_1to2 = np.array(
                [e[f'x{j + 1}'] - e[f'x{i + 1}'], e[f'y{j + 1}'] - e[f'y{i + 1}'], e[f'z{j + 1}'] - e[f'z{i + 1}']])
            vec_2to3 = np.array(
                [e[f'x{k + 1}'] - e[f'x{j + 1}'], e[f'y{k + 1}'] - e[f'y{j + 1}'], e[f'z{k + 1}'] - e[f'z{j + 1}']])
            theta2 = angle_between(vec_1to2, vec_2to3)

            x1 = e[f'x{i + 1}']
            y1 = e[f'y{i + 1}']
            z1 = e[f'z{i + 1}']

            # And now the incident gamma energy and theta1
            dE1 = e[f'dE{i + 1}']
            dE2 = e[f'dE{j + 1}']
            Eg = dE1 + 0.5 * (dE2 + np.sqrt(dE2 ** 2 + (4 * dE2 * me / (1 - np.cos(theta2)))))
            Egp = Eg - dE1
            theta1 = np.arccos(1 + me * ((1 / Eg) - (1 / Egp)))

            if np.isnan(theta1) or theta1 < theta1_min:
                # reject bad theta
                # print(theta1, me*( (1/Eg) - (1/Egp) ))
                elim_arrs.append(arr)
                continue

            dir_vec = unit_vector(np.array(
                [e[f'x{i + 1}'] - e[f'x{j + 1}'], e[f'y{i + 1}'] - e[f'y{j + 1}'], e[f'z{i + 1}'] - e[f'z{j + 1}']]))

            keep_arrs.append(arr)
            kept_theta1s.append(theta1)
            kept_dir_vecs.append(dir_vec)
            kept_s1_locs.append([x1, y1, z1])

            gsl = np.array(guess_source_loc)
            guess_source_vec = unit_vector(np.array([gsl[0] - x1, gsl[1] - y1, gsl[2] - z1]))
            ang_between_dir_and_guess = angle_between(dir_vec, guess_source_vec)
            kept_ang_between_dir_and_guess.append(ang_between_dir_and_guess)

            # print()

            kept_thetacalc_thetaguess_absdiffs.append(abs(theta1 - ang_between_dir_and_guess))

        if len(kept_theta1s) == 0:
            return False

        # Now find best of the non-culled arrangements
        # print(kept_thetacalc_thetaguess_absdiffs)
        i_best = kept_thetacalc_thetaguess_absdiffs.index(min(kept_thetacalc_thetaguess_absdiffs))

        theta = kept_theta1s[i_best]
        ang_between_dir_and_guess = kept_ang_between_dir_and_guess[i_best]
        s1_loc = kept_s1_locs[i_best]
        dir_vec = kept_dir_vecs[i_best]

        if reject_theta_diff_thresh == None:
            pass
        elif reject_theta_diff_thresh >= 0:  # absolute difference threshold, in radians
            diff = abs(theta - ang_between_dir_and_guess)
            if diff > reject_theta_diff_thresh:
                return False
        else:
            diff = abs(theta - ang_between_dir_and_guess) / ang_between_dir_and_guess
            if diff > abs(reject_theta_diff_thresh):
                return False

        # while troubleshooting, first just require it to match known MC truth
        if i_best != 0:
            pass
            #print(i_best)
            #print(kept_theta1s)
            ###print('theta_calc=', theta, ', theta_MC=', e['theta1_MCtruth'], ', dE1=', e['dE1'], ', dE2=', e['dE2'])
            #print('theta_calc=', theta, ', theta_MC=', 'UNKNOWN', ', dE1=', e['dE1'], ', dE2=', e['dE2'])

        # Force MC Truth answers
        # theta = e['theta1_MCtruth']
        # s1_loc = [e['x1'], e['y1'], e['z1']]
        # MCtruth_dir_vec = unit_vector(np.array([e['x1'] - e['x2'], e['y1'] - e['y2'], e['z1'] - e['z2']]))
        # MCtruth_dir_vec = unit_vector(np.array([e['x2'] - e['x1'], e['y2'] - e['y1'], e['z2'] - e['z1']]))
        # dir_vec = MCtruth_dir_vec
        # print('theta_calc=',theta,', theta_MC=',e['theta1_MCtruth'],', dE1=',e['dE1'],', dE2=',e['dE2'])
        # print(s1_loc,[e['x1'], e['y1'], e['z1']])
        # print(dir_vec,MCtruth_dir_vec)

        if return_only_best_cone:
            cone = np.empty((1), dtype=event_cone_type)
            cone['x'][0] = s1_loc[0]
            cone['y'][0] = s1_loc[1]
            cone['z'][0] = s1_loc[2]
            cone['u'][0] = dir_vec[0]
            cone['v'][0] = dir_vec[1]
            cone['w'][0] = dir_vec[2]
            cone['theta'][0] = theta
            cone['theta_MCtruth'][0] = 0 # e['theta1_MCtruth']

            return [cone]
        else:
            cones_list = []
            for ikc in range(len(kept_theta1s)):
                cone = np.empty((1), dtype=event_cone_type)
                s1_loc = kept_s1_locs[ikc]
                dir_vec = kept_dir_vecs[ikc]
                theta = kept_theta1s[ikc]
                cone['x'][0] = s1_loc[0]
                cone['y'][0] = s1_loc[1]
                cone['z'][0] = s1_loc[2]
                cone['u'][0] = dir_vec[0]
                cone['v'][0] = dir_vec[1]
                cone['w'][0] = dir_vec[2]
                cone['theta'][0] = theta
                cone['theta_MCtruth'][0] = 0 # e['theta1_MCtruth']
                cones_list.append(cone)
            return cones_list


    if make_gamma_cones:
        bad_g_events_supplied = 0
        i_ggc = 0

        print("\nBuilding gamma cones...   ({:0.2f} seconds elapsed)".format(time.time() - start))
        for i in range(num_gamma_records):
            this_event_record = gamma_records[i_gc]

            this_cone_list = build_gamma_cone(this_event_record, image_plane_norm_vec=plane_normal,
                                              guess_source_loc=source_coordinates,
                                              return_only_best_cone=True,
                                              reject_theta_diff_thresh=reject_theta_diff_thresh)
            if this_cone_list:
                for this_cone in this_cone_list:
                    g_cones[i_ggc] = this_cone
                    i_ggc += 1
                    indices_of_kept_event_records_g.append(i_gc)
                    if use_fastmode and i_ggc>fast_max_num_imaged_cones: break
            else:
                bad_g_events_supplied += 1

            i_gc += 1

        num_good_g_cones = i_ggc

        if num_good_g_cones==0:
            image_gamma_events = False

        g_cones = g_cones[:i_ggc]

        print(bad_g_events_supplied, 'bad gamma events supplied')
        print(num_good_g_cones, 'total good gamma cones to image')

        theta_g_calc_list = []
        theta_g_MCtruth_list = []
        for ic in range(len(g_cones)):
            theta_g_calc_list.append(g_cones['theta'][ic])
            theta_g_MCtruth_list.append(g_cones['theta_MCtruth'][ic])

        num_cones = len(theta_g_calc_list)

        title_str = r'$\theta$ calculated vs MC truth'
        fig1theta = plt.figure(figi)
        ax1theta = fig1theta.add_subplot(1, 1, 1, rasterized=True)
        fig1theta, ax1theta = fancy_plot(
            [theta_g_calc_list],
            [theta_g_MCtruth_list],
            x_scale='linear',
            y_scale='linear',
            x_label_str=r'$\theta$ calculated [radians]',
            y_label_str=r'$\theta$ MC truth [radians]',
            title_str=title_str,
            # marker='.',
            # linestyle='-',
            color='k',
            alpha=0.2,
            x_limits=[0, 1.1 * np.pi],
            y_limits=[0, 1.1 * np.pi],
            fig_width_inch=6.5,
            fig_height_inch=4.5,
            figi=figi,
            fig=fig1theta,
            ax=ax1theta
        )
        figi += 1
        # plt.xticks(np.arange(0, np.pi+np.pi/4, step=(np.pi/4)), ['0','π/4','π/2','3π/4','π'])
        # plt.yticks(np.arange(0, np.pi+np.pi/4, step=(np.pi/4)), ['0','π/4','π/2','3π/4','π'])
        axes = plt.gca()
        slx, sly = 0.5, 0.98
        text_on_plot = r'{:g} $\gamma$-ray event cones'.format(num_cones)
        axes.text(slx, sly, text_on_plot, color='black', horizontalalignment='center', verticalalignment='top',
                  transform=axes.transAxes, fontsize=16)  # , weight='bold')
        if save_plots:
            for ext in image_extensions:
                plot_save_path = images_path + 'gamma_' + MCtrutxt + slugify(
                    title_str) + ext  # or use fig.canvas.get_window_title()
                fig1theta.savefig(plot_save_path, facecolor=(0, 0, 0, 0), dpi=500, bbox_inches='tight')

        # plt.show()
        # sys.exit()

    results_dict.cones = Munch({
        'n_cones': n_cones,
        'g_cones': g_cones
    })


'''
██ ███    ███  █████   ██████  ██ ███    ██  ██████  
██ ████  ████ ██   ██ ██       ██ ████   ██ ██       
██ ██ ████ ██ ███████ ██   ███ ██ ██ ██  ██ ██   ███ 
██ ██  ██  ██ ██   ██ ██    ██ ██ ██  ██ ██ ██    ██ 
██ ██      ██ ██   ██  ██████  ██ ██   ████  ██████  
'''
#

temp_pickle_file = 'temp.pickle'
if __name__ == '__main__':
    # make temporary pickle object with just info needed by parallel
    # running_sums = [summed_image, sum_h, sum_h2, sum_v, sum_v2, num_pixel_hits]
    pickle_obj = [img_blc, i_match, non_match_i, hbin_mids, vbin_mids, hbin_edges, vbin_edges, num_hbins, num_vbins,
                  save_list_mode_results]
    make_pickle(temp_pickle_file, pickle_obj)
else:
    # load pickle with other variables needed for parallel processes
    p = read_pickle(temp_pickle_file)
    img_blc, i_match, non_match_i, hbin_mids, vbin_mids, hbin_edges, vbin_edges, num_hbins, num_vbins, save_list_mode_results = \
    p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    i_event_type = 0  # hard coded for neutrons only for now...
    # running sums
    summed_image = np.zeros((3, num_hbins, num_vbins))  # first index is 0=n+g cones, 0=n only cones, 1=g only cones
    # running sums of coordinate values for sake of variance calculation
    sum_h, sum_h2 = 0, 0
    sum_v, sum_v2 = 0, 0
    num_pixel_hits = 0

if __name__ == '__main__':
    # Functions for ray tracing approach
    def rotate_vecA_about_vecB_by_ang(A, B, theta):
        # based off of this: https://math.stackexchange.com/questions/511370/how-to-rotate-one-vector-about-another
        # which employs:https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        A = np.array(A)
        B = np.array(B)
        A_parB = (np.dot(A, B) / np.dot(B, B)) * B
        A_perpB = A - A_parB
        W = np.cross(B, A_perpB)
        mag_A_perpB = magnitude(A_perpB)
        x1 = np.cos(theta) / mag_A_perpB
        x2 = np.sin(theta) / magnitude(W)
        A_perpB_theta = mag_A_perpB * (x1 * A_perpB + x2 * W)
        A_rotated = A_perpB_theta + A_parB
        return A_rotated


    def rotate_vecA_inZ_by_ang(U, theta):
        # see: https://stackoverflow.com/questions/72374034/generating-two-vectors-with-a-given-angle-between-them
        U = np.array(U)
        Uhat = U / magnitude(U)
        # A = np.array([0,0,0])
        B = np.array([U[0], U[1], 0])
        H = B - np.dot(B, Uhat * Uhat)
        Hhat = H / magnitude(H)
        V = np.cos(theta) * Uhat + np.sin(theta) * Hhat
        return V


    def find_vector_plane_intersect(o, d, a, b, c):
        # Input position coordinates o, direction vector d,
        # and three points on plane
        # see: https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
        o, d, a, b, c = np.array(o), np.array(d), np.array(a) / magnitude(a), np.array(b) / magnitude(b), np.array(
            c) / magnitude(c)
        n = np.cross(a - b, b - c)
        t = ((n[0] * a[0] + n[1] * a[1] + n[2] * a[2]) - (n[0] * o[0] + n[1] * o[1] + n[2] * o[2])) / (
                    n[0] * d[0] + n[1] * d[1] + n[2] * d[2])
        intersect = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t]
        return intersect


    def find_vector_axis_plane_intersect(o, d, a, b, c, i_norm_ax):
        # Input position coordinates o, direction vector d,
        # and three points on plane, specifying which axis the plane is normal to: X (0), Y (1), or Z (2)
        # see: https://math.stackexchange.com/questions/100439/determine-where-a-vector-will-intersect-a-plane
        # This is faster than the generalized solution since the expression for t is much simpler
        o, d, a, b, c = np.array(o), np.array(d), np.array(a), np.array(b), np.array(c)
        n = np.cross(a - b, b - c)
        t = (n[i_norm_ax] * a[i_norm_ax] - n[i_norm_ax] * o[i_norm_ax]) / (n[i_norm_ax] * d[i_norm_ax])
        intersect = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t]
        return intersect


# Functions for matrix math approach
# Calculate cone equation
# https://math.stackexchange.com/questions/1834068/general-equation-of-a-cone
def calc_cone_matrix_M(d, theta):
    Dhat = np.asmatrix(unit_vector(d))
    M = (Dhat.T @ Dhat) - (np.cos(theta) ** 2) * np.identity(3)
    return M


def find_conic_roots_in_1D(vars=[None, None, None], o=[0, 0, 0], M=None, d=None, theta=None):
    '''
    Input:
     - vars = length 3 list containing x, y, and z. Two of these should be assigned float values while
              the third is set to None as will be the "unknown" variable solved for
     - o = vertex of cone (len 3 list of floats)
     - M = cone matrix (3x3 matrix), if not already calculated input:
        - d = direction vector
        - theta = cone half-angle
    '''
    if isinstance(M, type(None)):
        if isinstance(d, type(None)) or isinstance(theta, type(None)):
            print('"d" and "theta" must be provided to calculate "M" if "M" is not directly supplied')
            sys.exit()
        M = calc_cone_matrix_M(d, theta)
    if vars.count(None) != 1:
        print(
            '"vars" list should contain 2 floats and 1 None (denoting the unknown variable), arranged as [x,y,z],\nthe following was supplied: ',
            vars)
    i_unknown_var = vars.index(None)

    if i_unknown_var == 0:
        # if y and z are fixed, solve for ux
        uy = vars[1] - o[1]
        uz = vars[2] - o[2]
        a = M[0, 0]
        b = uy * (M[1, 0] + M[0, 1]) + uz * (M[2, 0] + M[0, 2])
        c = (uy ** 2) * M[1, 1] + (uy * uz) * (M[2, 1] + M[1, 2]) + (uz ** 2) * M[2, 2]
        u_add = o[0]
    elif i_unknown_var == 1:
        # if x and z are fixed, solve for uy
        ux = vars[0] - o[0]
        uz = vars[2] - o[2]
        a = M[1, 1]
        b = ux * (M[1, 0] + M[0, 1]) + uz * (M[2, 1] + M[1, 2])
        c = (ux ** 2) * M[0, 0] + (ux * uz) * (M[2, 0] + M[0, 2]) + (uz ** 2) * M[2, 2]
        u_add = o[1]
    elif i_unknown_var == 2:
        # if x and y are fixed, solve for uz
        ux = vars[0] - o[0]
        uy = vars[1] - o[1]
        a = M[2, 2]
        b = ux * (M[2, 0] + M[0, 2]) + uy * (M[2, 1] + M[1, 2])
        c = (ux ** 2) * M[0, 0] + (ux * uy) * (M[1, 0] + M[0, 1]) + (uy ** 2) * M[1, 1]
        u_add = o[2]
    else:
        print('invalid value of i_unknown_var entered, quitting...')
        sys.exit()
    p = [a, b, c]  # coefficients polynomial
    roots = u_add + np.roots(p)
    roots = roots[np.isreal(roots)]  # only return real roots, discarding complex values
    return roots


def init_globals(counter_num_pixel_hits, counter_sum_h, counter_sum_h2, counter_sum_v, counter_sum_v2):
    global _COUNTER_num_pixel_hits, _COUNTER_sum_h, _COUNTER_sum_h2, _COUNTER_sum_v, _COUNTER_sum_v2
    _COUNTER_num_pixel_hits = counter_num_pixel_hits
    _COUNTER_sum_h = counter_sum_h
    _COUNTER_sum_h2 = counter_sum_h2
    _COUNTER_sum_v = counter_sum_v
    _COUNTER_sum_v2 = counter_sum_v2


def image_cone(cone):
    global summed_image, sum_h, sum_h2, sum_v, sum_v2, num_pixel_hits
    o = [cone['x'], cone['y'], cone['z']]  # cone origin/apex
    d = [cone['u'], cone['v'], cone['w']]  # cone axis direction
    theta = cone['theta']  # cone angle
    M = calc_cone_matrix_M(d, theta)
    plotx = []
    ploty = []
    local_sum_h = 0
    local_sum_h2 = 0
    local_sum_v = 0
    local_sum_v2 = 0
    local_num_pixel_hits = 0
    # Now scan horizontal pixel rows
    vars = [None, None, None]
    vars[i_match] = img_blc[i_match]
    for bin_mid in hbin_mids:
        vars[non_match_i[0]] = bin_mid
        try:
            roots = find_conic_roots_in_1D(vars=vars, o=o, M=M)
        except:
            continue
        for i in range(len(roots)):
            plotx.append(bin_mid)
            ploty.append(roots[i])
            if roots[i] >= vbin_edges[0] and roots[i] <= vbin_edges[-1]:
                local_sum_h += bin_mid
                local_sum_h2 += bin_mid ** 2
                local_sum_v += roots[i]
                local_sum_v2 += roots[i] ** 2
                local_num_pixel_hits += 1

                # with _COUNTER_num_pixel_hits.get_lock(): _COUNTER_num_pixel_hits.value += 1
                # with _COUNTER_sum_h.get_lock(): _COUNTER_sum_h.value += bin_mid
                # with _COUNTER_sum_h2.get_lock(): _COUNTER_sum_h2.value += bin_mid**2
                # with _COUNTER_sum_v.get_lock(): _COUNTER_sum_v.value += roots[i]
                # with _COUNTER_sum_v2.get_lock(): _COUNTER_sum_v2.value += roots[i]**2

    # Now scan vertical pixel columns
    vars = [None, None, None]
    vars[i_match] = img_blc[i_match]
    for bin_mid in vbin_mids:
        vars[non_match_i[1]] = bin_mid
        try:
            roots = find_conic_roots_in_1D(vars=vars, o=o, M=M)
        except:
            continue
        for i in range(len(roots)):
            plotx.append(roots[i])
            ploty.append(bin_mid)
            if roots[i] >= hbin_edges[0] and roots[i] <= hbin_edges[-1]:
                local_sum_h += roots[i]
                local_sum_h2 += roots[i] ** 2
                local_sum_v += bin_mid
                local_sum_v2 += bin_mid ** 2
                local_num_pixel_hits += 1

    with _COUNTER_num_pixel_hits.get_lock():
        _COUNTER_num_pixel_hits.value += local_num_pixel_hits
    with _COUNTER_sum_h.get_lock():
        _COUNTER_sum_h.value += local_sum_h
    with _COUNTER_sum_h2.get_lock():
        _COUNTER_sum_h2.value += local_sum_h2
    with _COUNTER_sum_v.get_lock():
        _COUNTER_sum_v.value += local_sum_v
    with _COUNTER_sum_v2.get_lock():
        _COUNTER_sum_v2.value += local_sum_v2

    intersect_hist, xedges, yedges = np.histogram2d(plotx, ploty, bins=[hbin_edges, vbin_edges])
    # intersect_hist = intersect_hist.T
    intersect_hist[intersect_hist > 0] = 1

    # summed_image[0,:,:] = summed_image[0,:,:] + intersect_hist
    # summed_image[1+i_event_type,:,:] = summed_image[1+i_event_type,:,:] + intersect_hist

    # print('in function: ',num_pixel_hits)

    # print(np.shape(intersect_hist),np.sum(intersect_hist))
    if save_list_mode_results:
        packedbits_image = np.packbits(np.array(intersect_hist, dtype=bool))
        return packedbits_image
        # imaged_cones_packedbits_list.append(packedbits_image)
    else:
        return


if __name__ == '__main__':
    # source coordinates defined in SETTINGS section now
    # source_coordinates = [0,0,0]
    # source_coordinates = [90,0,0]

    xdata_lists = []
    ydata_lists = []

    theta_calc_list = []
    theta_MCtruth_list = []

    summed_image = np.zeros((3, num_hbins, num_vbins))  # first index is 0=n+g cones, 0=n only cones, 1=g only cones

    # running sums of coordinate values for sake of variance calculation
    list_sum_h, list_sum_h2 = [0,0,0], [0,0,0]
    list_sum_v, list_sum_v2 = [0,0,0], [0,0,0]
    list_num_pixel_hits = [0,0,0]



    # if make_neutron_cones and image_neutron_events:

    par_cones_list = [n_cones, g_cones]
    par_types_str = ['neutron', 'gamma']

    num_par_cones = [0,0,0]

    imaged_cones_packedbits_list = []
    imaged_cones_list_of_packedbits_lists = [[],[]]

    for i_event_type in range(len(par_cones_list)):
        par_cones = par_cones_list[i_event_type]
        if i_event_type == 0 and not (make_neutron_cones and image_neutron_events):
            print('Not imaging neutron cones...')
            continue
        if i_event_type == 1 and not (make_gamma_cones and image_gamma_events):
            print('Not imaging gamma cones...')
            continue
        print("\nImaging {:} cones...   ({:0.2f} seconds elapsed)".format(par_types_str[i_event_type],
                                                                          time.time() - start))

        sum_h, sum_h2 = 0, 0
        sum_v, sum_v2 = 0, 0
        num_pixel_hits = 0

        if use_parallelization_for_imaging:
            counter_num_pixel_hits = mp.Value('i', 0)
            counter_sum_h = mp.Value('d', 0)
            counter_sum_h2 = mp.Value('d', 0)
            counter_sum_v = mp.Value('d', 0)
            counter_sum_v2 = mp.Value('d', 0)

        if i_event_type == 0:
            num_good_cones = num_good_n_cones
        else:
            num_good_cones = num_good_g_cones

        if use_parallelization_for_imaging:

            '''
            def main():
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                    imaged_cones_packedbits_list = executor.map(image_cone,par_cones)
                    print(type(imaged_cones_packedbits_list))

            if __name__ == '__main__':
                main()
            '''

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes, initializer=init_globals, initargs=(
            counter_num_pixel_hits, counter_sum_h, counter_sum_h2, counter_sum_v,
            counter_sum_v2)) as executor:  # ThreadPoolExecutor
                results = executor.map(image_cone, par_cones)
                imaged_cones_packedbits_list = [result for result in results]
                imaged_cones_list_of_packedbits_lists[i_event_type] = imaged_cones_packedbits_list
                if print_debug_statements: print(type(imaged_cones_packedbits_list))
            print(
                '\tImaged all of {:g} cones...   ({:0.2f} seconds elapsed)'.format(num_good_cones, time.time() - start))

            num_pixel_hits = counter_num_pixel_hits.value
            sum_h = counter_sum_h.value
            sum_h2 = counter_sum_h2.value
            sum_v = counter_sum_v.value
            sum_v2 = counter_sum_v2.value

            list_sum_h[0] += sum_h
            list_sum_h[1 + i_event_type] += sum_h
            list_sum_h2[0] += sum_h2
            list_sum_h2[1 + i_event_type] += sum_h2
            list_sum_v[0] += sum_v
            list_sum_v[1 + i_event_type] += sum_v
            list_sum_v2[0] += sum_v2
            list_sum_v2[1 + i_event_type] += sum_v2
            list_num_pixel_hits[0] += num_pixel_hits
            list_num_pixel_hits[1 + i_event_type] += num_pixel_hits

            # update running sums with temp files
            # find running sum temp files
            '''
            running_sum_files = []
            for f in os.listdir():
                if 'temp_running_sums_pid' in f:
                    running_sum_files.append(f)
            # add running sums
            for f in running_sum_files:
                running_sums = read_pickle(f)
                summed_image += running_sums[0]
                sum_h += running_sums[1]
                sum_h2 += running_sums[2]
                sum_v += running_sums[3]
                sum_v2 += running_sums[4]
                num_pixel_hits += running_sums[5]
            # delete temp pickle files
            for f in running_sum_files:
                os.remove(f)
            '''

            '''
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                imaged_cones_packedbits_list = executor.map(image_cone,par_cones)
                print(type(imaged_cones_packedbits_list))
                #imaged_cones_packedbits_list = results
            '''

            # for ic in range(len(par_cones)):
            #    theta_calc_list.append(par_cones['theta'][ic])
            #    theta_MCtruth_list.append(par_cones['theta_MCtruth'][ic])


        else:  # single-threaded imaging
            for ic in range(len(par_cones)):
                if max_num_cones != None:
                    if ic > max_num_cones: break  # limit it just to not print too much while debugging
                # ic=6 # in Lena data, only bottom half of elipse in imaging plane
                # ic=1 # in Lena data, whole elipse in imaging plane
                # ic=2 # in Lena data, none in imaging plane
                # ic=7
                if ic % 1000 == 0:
                    print('\tImaged {:g} of {:g} cones...   ({:0.2f} seconds elapsed)'.format(ic, num_good_cones,
                                                                                              time.time() - start))
                num_par_cones[0] += 1
                num_par_cones[1+i_event_type] += 1

                o = [par_cones['x'][ic], par_cones['y'][ic], par_cones['z'][ic]]  # cone origin/apex
                d = [par_cones['u'][ic], par_cones['v'][ic], par_cones['w'][ic]]  # cone axis direction
                theta = par_cones['theta'][ic]  # cone angle
                # theta = n_cones['theta_MCtruth'][ic] # cone angle
                # print(n_cones['theta'][ic],n_cones['theta_MCtruth'][ic])

                theta_calc_list.append(par_cones['theta'][ic])
                theta_MCtruth_list.append(par_cones['theta_MCtruth'][ic])
                # print(d,theta*180/np.pi)

                if image_via_matrix_math:
                    # calculate cone matrix (only needed once per cone)
                    M = calc_cone_matrix_M(d, theta)
                    # print(M)

                    plotx = []
                    ploty = []

                    # Now scan horizontal pixel rows
                    vars = [None, None, None]
                    vars[i_match] = img_blc[i_match]
                    for bin_mid in hbin_mids:
                        vars[non_match_i[0]] = bin_mid
                        try:
                            roots = find_conic_roots_in_1D(vars=vars, o=o, M=M)
                        except:
                            continue
                        for i in range(len(roots)):
                            plotx.append(bin_mid)
                            ploty.append(roots[i])
                            if roots[i] >= vbin_edges[0] and roots[i] <= vbin_edges[-1]:
                                sum_h += bin_mid
                                sum_h2 += bin_mid ** 2
                                sum_v += roots[i]
                                sum_v2 += roots[i] ** 2
                                num_pixel_hits += 1

                    # Now scan vertical pixel columns
                    vars = [None, None, None]
                    vars[i_match] = img_blc[i_match]
                    for bin_mid in vbin_mids:
                        vars[non_match_i[1]] = bin_mid
                        try:
                            roots = find_conic_roots_in_1D(vars=vars, o=o, M=M)
                        except:
                            continue
                        for i in range(len(roots)):
                            plotx.append(roots[i])
                            ploty.append(bin_mid)
                            if roots[i] >= hbin_edges[0] and roots[i] <= hbin_edges[-1]:
                                sum_h += roots[i]
                                sum_h2 += roots[i] ** 2
                                sum_v += bin_mid
                                sum_v2 += bin_mid ** 2
                                num_pixel_hits += 1

                    intersect_hist, xedges, yedges = np.histogram2d(plotx, ploty, bins=[hbin_edges, vbin_edges])
                    # intersect_hist = intersect_hist.T
                    intersect_hist[intersect_hist > 0] = 1

                    # print(np.shape(intersect_hist),np.sum(intersect_hist))
                    if save_list_mode_results:
                        packedbits_image = np.packbits(np.array(intersect_hist, dtype=bool))
                        imaged_cones_packedbits_list.append(packedbits_image)

                    # unpacked = np.unpackbits(packedbits_image)[:intersect_hist.size].reshape(intersect_hist.shape).view(bool)
                    # print(np.shape(unpacked), np.sum(unpacked))
                    # sys.exit()

                    summed_image[0, :, :] = summed_image[0, :, :] + intersect_hist
                    summed_image[1 + i_event_type, :, :] = summed_image[1 + i_event_type, :, :] + intersect_hist



                    # xdata_lists.append(plotx)
                    # ydata_lists.append(ploty)

                if image_via_ray_tracing:
                    # for debugging, manually feed a cone
                    # o = [100,0,0]
                    # d = [-1,0,0]
                    # theta = (5.0)*np.pi/180

                    # Filter out cones resulting in parabolas or hyperbolas
                    # ang_between_d_and_plane_norm = angle_between(d, -1 * plane_normal, degrees=False)
                    # max_ang_wrt_plane = ang_between_d_and_plane_norm + theta
                    # print(ang_between_d_and_plane_norm*180/np.pi , theta*180/np.pi)
                    # print(max_ang_wrt_plane*180/np.pi)
                    # if max_ang_wrt_plane >= np.pi/2:
                    #    continue

                    # Find a vector along cone edge (with theta between it and cone axis)
                    # lazy way to do this is to fix x and y of new vector to be same as axis and find z for new angle
                    Vstart = rotate_vecA_inZ_by_ang(d, theta)

                    # Rotate cone edge vector about cone axis in delta_phi steps in [0,2pi]
                    # and find intersection point with imaging plane
                    phi_start = 0
                    phi_stop = 3 * np.pi
                    min_phi_imaged = 3 * np.pi
                    max_phi_imaged = 0
                    n_phi_first_pass = 150
                    n_phi = 60

                    hbin_min = hbin_edges[0]
                    hbin_max = hbin_edges[-1]
                    vbin_min = vbin_edges[0]
                    vbin_max = vbin_edges[-1]

                    min_h_imaged = hbin_max
                    max_h_imaged = hbin_min
                    min_v_imaged = vbin_max
                    max_v_imaged = vbin_min

                    consecutive_phi_this = []
                    consecutive_phi_record = []
                    on_a_streak = False

                    if iteratively_search_for_cone_image_bounds:
                        # Iteratively find what range of phi values intersect with imaging plane
                        n_iter_max = 3
                        n_iter = n_iter_max
                        while n_iter > 0:
                            if n_iter == n_iter_max:
                                n_phi_this = n_phi_first_pass
                            else:
                                n_phi_this = n_phi
                            dphi = (phi_stop - phi_start) / n_phi_this
                            phi_list = np.linspace(phi_start, phi_stop, num=n_phi_this, endpoint=False)
                            min_phi_imaged_old = min_phi_imaged
                            max_phi_imaged_old = max_phi_imaged
                            for i_a, phi in enumerate(phi_list):
                                angle = phi_list[i_a]
                                V_cone_edge_i = rotate_vecA_about_vecB_by_ang(Vstart, d, angle)
                                intersect = find_vector_axis_plane_intersect(o, V_cone_edge_i, pa, pb, pc, i_match)
                                if (intersect[non_match_i[0]] > hbin_min) and (intersect[non_match_i[0]] < hbin_max) and \
                                        (intersect[non_match_i[1]] > vbin_min) and (
                                        intersect[non_match_i[1]] < vbin_max):
                                    # ray intersects imaging plane
                                    on_a_streak = True
                                    # print(phi,i_a)
                                    consecutive_phi_this.append(phi)
                                    # if intersect[non_match_i[0]] < min_h_imaged: min_h_imaged = intersect[non_match_i[0]] < min_h_imaged
                                    # if phi < min_phi_imaged: min_phi_imaged = phi
                                    # if phi > max_phi_imaged: max_phi_imaged = phi
                                else:
                                    if on_a_streak:  # now streak is over
                                        if len(consecutive_phi_this) > len(consecutive_phi_record):
                                            consecutive_phi_record = consecutive_phi_this
                                        consecutive_phi_this = []
                                    on_a_streak = False
                            # print(consecutive_phi_record)
                            if consecutive_phi_record == []:  # none hit imaging plane
                                break
                            min_phi_imaged = min(consecutive_phi_record)
                            max_phi_imaged = max(consecutive_phi_record)

                            phi_start_new = max(phi_start, min_phi_imaged - dphi)
                            phi_stop_new = min(phi_stop, max_phi_imaged + dphi)
                            # print(phi_start,phi_stop,phi_start_new,phi_stop_new,min_phi_imaged,max_phi_imaged)
                            if (phi_start_new == phi_start) and (phi_stop_new == phi_stop):
                                break  # whole [0,2pi] range inside imaging window
                            # if (min_phi_imaged==min_phi_imaged_old) and (max_phi_imaged==max_phi_imaged_old):
                            #    break # none of the sampled [0,2pi] range was inside imaging window
                            n_iter += -1
                            phi_start = phi_start_new
                            phi_stop = phi_stop_new
                    else:
                        phi_start = 0
                        phi_stop = 2 * np.pi

                    n_phi = 360
                    phi_list = np.linspace(phi_start, phi_stop, num=n_phi, endpoint=False)
                    plotx = []
                    ploty = []
                    for i_a, phi in enumerate(phi_list):
                        # sample rotated vector
                        angle = phi_list[i_a]
                        V_cone_edge_i = rotate_vecA_about_vecB_by_ang(Vstart, d, angle)
                        intersect = find_vector_axis_plane_intersect(o, V_cone_edge_i, pa, pb, pc, i_match)
                        plotx.append(intersect[1])
                        ploty.append(intersect[2])
                        # print('{:g} \t{:g}'.format(intersect[1],intersect[2]))

                    intersect_hist, xedges, yedges = np.histogram2d(plotx, ploty, bins=[hbin_edges,
                                                                                        vbin_edges])  # bins=[num_hbins,num_vbins],range=[im_lr_range,im_bt_range])
                    # intersect_hist = intersect_hist.T
                    intersect_hist[intersect_hist > 0] = 1

                    summed_image[0, :, :] = summed_image[0, :, :] + intersect_hist
                    summed_image[1 + i_event_type, :, :] = summed_image[1 + i_event_type, :, :] + intersect_hist

                    xdata_lists.append(plotx)
                    ydata_lists.append(ploty)

            list_sum_h[0] += sum_h
            list_sum_h[1 + i_event_type] += sum_h
            list_sum_h2[0] += sum_h2
            list_sum_h2[1 + i_event_type] += sum_h2
            list_sum_v[0] += sum_v
            list_sum_v[1 + i_event_type] += sum_v
            list_sum_v2[0] += sum_v2
            list_sum_v2[1 + i_event_type] += sum_v2
            list_num_pixel_hits[0] += num_pixel_hits
            list_num_pixel_hits[1 + i_event_type] += num_pixel_hits
            imaged_cones_list_of_packedbits_lists[i_event_type] = imaged_cones_packedbits_list
            #imaged_cones_list_of_packedbits_lists.append(imaged_cones_packedbits_list)
else:
    # print('in script after function: ',num_pixel_hits)
    # running_sums = [summed_image, sum_h, sum_h2, sum_v, sum_v2, num_pixel_hits]
    # running_sums_pickle_name = 'temp_running_sums_pid_{}.pickle'.format(os.getpid())
    # make_pickle(running_sums_pickle_name, running_sums)
    # print('\t\t\t({:0.2f} seconds elapsed on this pid)'.format(time.time() - start_process))
    print('\tStarting process id {}'.format(os.getpid()))

if __name__ == '__main__':

    if use_parallelization_for_imaging:  # load and sum temp pickles

        # print(counter)
        # num_pixel_hits = counter.value
        '''
        num_pixel_hits = counter_num_pixel_hits.value
        sum_h = counter_sum_h.value
        sum_h2 = counter_sum_h2.value
        sum_v = counter_sum_v.value
        sum_v2 = counter_sum_v2.value
        # print(num_pixel_hits)
        '''


        for i_event_type in range(len(par_cones_list)):
            if i_event_type == 0 and not (make_neutron_cones and image_neutron_events):
                print('Not imaging neutron cones...')
                continue
            if i_event_type == 1 and not (make_gamma_cones and image_gamma_events):
                print('Not imaging gamma cones...')
                continue
            print("\nAssembling packedbits image array from {:} cones...   ({:0.2f} seconds elapsed)".format(par_types_str[i_event_type], time.time() - start))
            par_cones = par_cones_list[i_event_type]

            # num_good_cones = len(image_indices[i_event_type]) # num_good_n_cones
            if i_event_type == 0:
                num_good_cones = num_good_n_cones
            else:
                num_good_cones = num_good_g_cones
            print(i_event_type,num_good_cones)
            # for ic in range(len(par_cones)):
            intersect_hist_size = (len(hbin_edges) - 1) * (len(vbin_edges) - 1)
            intersect_hist_shape = (len(hbin_edges) - 1, len(vbin_edges) - 1)
            for ic in range(num_good_cones):
                if max_num_cones != None:
                    if ic > max_num_cones: break  # limit it just to not print too much while debugging
                # ici=7
                if ic % 5000 == 0:
                    print('\tImaged {:g} of {:g} cones...   ({:0.2f} seconds elapsed)'.format(ic, num_good_cones,
                                                                                              time.time() - start))
                num_par_cones[0] += 1
                num_par_cones[1+i_event_type] += 1

                #packedbits_image = imaged_cones_packedbits_list[ic]
                packedbits_image = imaged_cones_list_of_packedbits_lists[i_event_type][ic]
                intersect_hist = np.unpackbits(packedbits_image)[:intersect_hist_size].reshape(
                    intersect_hist_shape).view(bool)
                # print(np.shape(unpacked), np.sum(unpacked))
                # sys.exit()

                summed_image[0, :, :] = summed_image[0, :, :] + intersect_hist
                summed_image[1 + i_event_type, :, :] = summed_image[1 + i_event_type, :, :] + intersect_hist
        print("\nDone processing packedbits image array...   ({:0.2f} seconds elapsed)".format(time.time() - start))

        '''
        running_sum_files = []
        for f in os.listdir():
            if 'temp_running_sums_pid' in f:
                running_sum_files.append(f)
        # add running sums
        for f in running_sum_files:
            running_sums = read_pickle(f)
            summed_image += running_sums[0]
            sum_h += running_sums[1]
            sum_h2 += running_sums[2]
            sum_v += running_sums[3]
            sum_v2 += running_sums[4]
            num_pixel_hits += running_sums[5]
        # delete temp pickle files
        for f in running_sum_files:
            os.remove(f)
        '''

    for ict in range(3): # ict > 0=n+g, 1=n, 2=g

        num_pixel_hits = list_num_pixel_hits[ict]
        sum_h = list_sum_h[ict]
        sum_h2 = list_sum_h2[ict]
        sum_v = list_sum_v[ict]
        sum_v2 = list_sum_v2[ict]

        if print_debug_statements: print(num_pixel_hits)

        # calculate true statistics
        N = num_pixel_hits
        if N==0:
            h_mean_true, v_mean_true = 0,0
            h_stdv_true, v_stdv_true = 0,0
            h_st_err_true, v_st_err_true = 0,0
        else:
            h_mean_true = sum_h / N
            v_mean_true = sum_v / N
            h_stdv_true = np.sqrt((N / (N - 1)) * ((sum_h2 / N) - h_mean_true ** 2))
            v_stdv_true = np.sqrt((N / (N - 1)) * ((sum_v2 / N) - v_mean_true ** 2))
            h_st_err_true = h_stdv_true / np.sqrt(N)
            v_st_err_true = v_stdv_true / np.sqrt(N)


        this_dict = Munch({
            'summed_image': summed_image, # currently just writing full array 3 times...
            'num_par_cones': num_par_cones[ict],
            'stats_true': Munch({
                'num_pixel_hits': num_pixel_hits,
                'h_mean_true': h_mean_true,
                'h_stdv_true': h_stdv_true,
                'h_st_err_true': h_st_err_true,
                'v_mean_true': v_mean_true,
                'v_stdv_true': v_stdv_true,
                'v_st_err_true': v_st_err_true
            })
        })

        if ict==0:
            results_dict.image = this_dict
        elif ict==1:
            results_dict.image.n = this_dict
        elif ict==2:
            results_dict.image.g = this_dict

    # Save list mode results: DAQ data, cones, and images (packbits)
    if save_list_mode_results:
        # kept_neutron_records = [neutron_records[i] for i in indices_of_kept_event_records_n]
        # kept_gamma_records = [gamma_records[i] for i in indices_of_kept_event_records_g]

        kept_neutron_records = np.empty((len(indices_of_kept_event_records_n)), dtype=neutron_event_record_type)
        for i in range(len(kept_neutron_records)):
            kept_neutron_records[i] = neutron_records[indices_of_kept_event_records_n[i]]

        kept_gamma_records = np.empty((len(indices_of_kept_event_records_g)), dtype=gamma_event_record_type)
        for i in range(len(kept_gamma_records)):
            kept_gamma_records[i] = gamma_records[indices_of_kept_event_records_g[i]]

        #print(len(imaged_cones_packedbits_list))
        if (make_neutron_cones and image_neutron_events):
            print("Size of neutron packedbits list of images: " + str(sys.getsizeof(imaged_cones_list_of_packedbits_lists[0])) + " bytes")
        if (make_gamma_cones and image_gamma_events):
            print("Size of gamma packedbits list of images: " + str(sys.getsizeof(imaged_cones_list_of_packedbits_lists[1])) + " bytes")
        #print(len(indices_of_kept_event_records_n))
        print("Size of neutron records list: " + str(sys.getsizeof(kept_neutron_records)) + " bytes")
        #print(len(indices_of_kept_event_records_g))
        #print(len(par_cones_list[0]))
        print("Size of neutron cones list: " + str(sys.getsizeof(par_cones_list[0])) + " bytes")
        print("Size of gamma records list: " + str(sys.getsizeof(kept_gamma_records)) + " bytes")
        print("Size of gamma cones list: " + str(sys.getsizeof(par_cones_list[1])) + " bytes")
        #print(len(par_cones_list[1]))
        '''
        list_mode_image_data_dict = Munch({
            'imaged_cones_packedbits_list':imaged_cones_packedbits_list,
            'n_cones_list':par_cones_list[0],
            'g_cones_list':par_cones_list[1],
            'kept_neutron_records':kept_neutron_records,
            'kept_gamma_records':kept_gamma_records,
            'meta':results_dict.meta,
        })
        '''
        print("\nSaving list mode results...   ({:0.2f} seconds elapsed)".format(time.time() - start))
        #temporary_disable_list_mode_pickle_save = True
        if temporary_disable_list_mode_pickle_save:
            print('\n','temporary_disable_list_mode_pickle_save = True')
            print('THEREFORE NOT WRITING Pickle file:', list_mode_image_results_pickle_path, '\n')
        else:
            with open(list_mode_image_results_pickle_path, 'wb') as handle:
                # to_be_pickled = results_dict
                pickle.dump(
                    Munch({
                        'imaged_n_cones_packedbits_list': imaged_cones_list_of_packedbits_lists[0],
                        'imaged_g_cones_packedbits_list': imaged_cones_list_of_packedbits_lists[1],
                        'n_cones_list': par_cones_list[0],
                        'g_cones_list': par_cones_list[1],
                        'kept_neutron_records': kept_neutron_records,
                        'kept_gamma_records': kept_gamma_records,
                        'meta': results_dict.meta,
                    }), handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('Pickle file written:', list_mode_image_results_pickle_path, '\n')

    print("\nGenerating plots...   ({:0.2f} seconds elapsed)".format(time.time() - start))

    if not using_experimental_data:
        if use_Lena_test_data:
            path_to_pickle_file = images_path + 'image_data.pickle'
            make_pickle(path_to_pickle_file, summed_image[0, :, :])
            # test = read_pickle(path_to_pickle_file)
            # print(np.shape(test))


    ng_tex_list = [r'n+$\gamma$','n',r'$\gamma$']
    ng_plaintxt_list = ['ng','n','g']
    for ict in range(3): # ict > 0=n+g, 1=n, 2=g

        #if ict==0 and not (image_neutron_events and image_gamma_events): continue
        if ict==1 and not image_neutron_events: continue
        if ict==2 and not image_gamma_events: continue

        ng_tex = ng_tex_list[ict]
        ng_plaintxt = ng_plaintxt_list[ict] + '_'

        W_cmap = 'cividis'

        x_label_str = '{:}-axis [mm]'.format(axes_list[non_match_i[0]])
        y_label_str = '{:}-axis [mm]'.format(axes_list[non_match_i[1]])
        z_label_str = r'counts per ${{{:g}}}\times{{{:g}}}$ mm$^2$ pixel'.format(10 * pixel_vwidth, 10 * pixel_hwidth)
        title_str = ng_tex+' cone intersections at {:}={:g} mm\n'.format(axes_list[i_match], 10 * img_blc[i_match]) + \
                    'source at ({:g},{:g},{:g}), '.format(10 * source_coordinates[0], 10 * source_coordinates[1],
                                                          10 * source_coordinates[2]) + \
                    'NOVO face at ({:g},0,0)'.format(10 * NOVO_face_x_coord)

        num_cones = num_par_cones[ict]

        plot_dict = Munch({
            'xdata': [10 * hbin_mids],
            'ydata': [10 * vbin_mids],
            'zdata': summed_image[ict, :, :],
            'x_label_str': x_label_str,
            'y_label_str': y_label_str,
            'z_label_str': z_label_str,
            'title_str': title_str,
        })
        if ict==0:
            results_dict.image.plot3D = plot_dict
        elif ict==1:
            results_dict.image.plot3D.n = plot_dict
        elif ict==2:
            results_dict.image.plot3D.g = plot_dict

        fig2, ax2 = fancy_3D_plot(
            [10 * hbin_mids],
            [10 * vbin_mids],
            summed_image[ict, :, :],
            plot_styles='map_pcolormesh',
            cmap=W_cmap,
            figi=figi,
            x_label_str=x_label_str,
            y_label_str=y_label_str,
            z_label_str=z_label_str,
            title_str=title_str,
            z_scale='linear',
            fig_width_inch=8,
            fig_height_inch=6
        )
        figi += 1
        fig2.tight_layout()

        axes = plt.gca()
        slx, sly = -0.6, 0.98  # grabs colorbar coords for whatever reason
        text_on_plot = r'{:g} {} event cones'.format(num_cones,ng_tex)  # $\gamma$-ray
        axes.text(slx, sly, text_on_plot, color='white', horizontalalignment='right', verticalalignment='top',
                  transform=axes.transAxes, fontsize=16)  # , weight='bold')

        if save_plots:
            for ext in image_extensions:
                plot_save_path = images_path + ng_plaintxt + 'image' + '_' + MCtrutxt[:-1] + ext  # or use fig.canvas.get_window_title()
                # plot_save_path = images_path + slugify(title_str) + MCtrutxt + ext # or use fig.canvas.get_window_title()
                fig2.savefig(plot_save_path, facecolor=(0, 0, 0, 0))

        # Generate plots projected onto each axis
        use_true_stats = True  # if True, use stats from roots, if False use stats from pixel data
        proj_peak_window_width_cm = 40  # 60 # this is used for finding the peak location in a rolling avg window of this width
        use_lmfit_peak_instead = True
        use_proj_peak_window = True
        proj_lmfit_peak_window_width_cm = 60  # 40
        num_fwhm_in_peak_fit = 0.28  # if using lmfit to find peak and not proj_peak_window, use this to determine width of peak region considered as a multiple of FWHM
        rebin_plotted_data = True  # change bin size of reults
        plot_bin_width_cm = 1  # rebin data to be plotted to have this bin width (to reduce noise)
        for ai in range(2):
            xdata_list = []
            ydata_list = []
            dist_yvals = []
            lmfit_peak_centers = []
            lmfit_peak_center_errs = []
            x_vals = [10 * hbin_mids, 10 * vbin_mids][ai]
            x_edges = [10 * hbin_edges, 10 * vbin_edges][ai]
            y_vals = np.sum(summed_image[ict, :, :], axis=1 - ai)
            if use_true_stats:
                mean_val = [h_mean_true, v_mean_true][ai] * 10
                stdv_val = [h_stdv_true, v_stdv_true][ai] * 10
                std_err_of_mean = [h_st_err_true, v_st_err_true][ai] * 10
            else:
                mean_val = np.sum(x_vals * y_vals) / np.sum(y_vals)
                stdv_val = np.sqrt(np.sum(y_vals * (x_vals - mean_val) ** 2) / (np.sum(y_vals) - 1))
                std_err_of_mean = stdv_val / np.sqrt(np.sum(y_vals))
            mean_str = r'$\overline{\mu}$' + '={:0.3g} mm'.format(
                mean_val) + '\n' + r'$\sigma_{\overline{\mu}}$' + '={:0.3g} mm'.format(
                std_err_of_mean) + '\n' + r'$\sigma$={:0.3g} mm'.format(stdv_val)
            x_str = [x_label_str, y_label_str][ai]
            y_str = r'counts per ${{{:g}}}$ mm bin'.format(10 * [pixel_vwidth, pixel_hwidth][ai])
            title_str = ng_tex + ' cone ints. at {:}={:g} mm, '.format(axes_list[i_match], 10 * img_blc[i_match]) + \
                        '{:} proj. over {:}[{:g},{:g}]\n'.format(axes_list[non_match_i[ai]], axes_list[non_match_i[1 - ai]],
                                                                 10 * [hbin_edges, vbin_edges][1 - ai][0],
                                                                 10 * [hbin_edges, vbin_edges][1 - ai][-1]) + \
                        'source at ({:g},{:g},{:g}), '.format(10 * source_coordinates[0], 10 * source_coordinates[1],
                                                              10 * source_coordinates[2]) + \
                        'NOVO face at ({:g},0,0)'.format(10 * NOVO_face_x_coord)
            plotx, ploty = generate_line_bar_coordinates(x_edges, y_vals)
            xdata_list.append(plotx)
            ydata_list.append(ploty)

            if ai == 0:
                pixel_width = results_dict.meta.pixel_vwidth
            else:
                pixel_width = results_dict.meta.pixel_hwidth

            if rebin_plotted_data:
                new_x_bins = np.arange(x_edges[0], x_edges[-1], plot_bin_width_cm * 10)
                new_y_bins = rebinner(new_x_bins, x_edges, y_vals)
                newx_bars, newy_bars = generate_line_bar_coordinates(new_x_bins, new_y_bins)
                indx = len(xdata_list) - 1
                xdata_list[indx] = newx_bars
                ydata_list[indx] = newy_bars

            if len(ydata_list) == 1:
                # find normalization factor, so peaks will hover around 1
                div_by_norm_factor = max(ydata_list[0])
            ydata_list[-1] = np.array(ydata_list[-1]) / div_by_norm_factor
            dist_yvals.append(ydata_list[-1])

            # use rolling window approach to find peak
            window_width_nbins = int(proj_peak_window_width_cm / pixel_width)  # 200 #35
            max_count = 0
            xavg_max_count = 0
            for j in range(len(y_vals) - window_width_nbins):
                this_sum = sum(y_vals[j:j + window_width_nbins])
                if this_sum > max_count:
                    max_count = this_sum
                    xavg_max_count = sum(
                        np.array(x_vals[j:j + window_width_nbins]) * np.array(y_vals[j:j + window_width_nbins])) / (
                                         sum(y_vals[j:j + window_width_nbins]))

            use_lmfit_peak_instead = True
            use_alternate_nonGauss_dist = True
            subtract_constant_background = True
            alt_model = 'Gaussian'
            if use_lmfit_peak_instead:
                # lmfit stuff
                if use_alternate_nonGauss_dist:
                    #                     0           1          2          3          x4                 x5                6             7           8           x9            10             11       12
                    alt_model_list = ['Lorentzian','Voigt','PseudoVoigt','Moffat','StudentsT','ExponentialGaussian','SplitLorentzian','Pearson4','Pearson7','BreitWigner','SkewedGaussian','Doniach','Lognormal']
                    alt_model = alt_model_list[8]
                    if alt_model=='Lorentzian':
                        from lmfit.models import LorentzianModel
                        model = LorentzianModel()
                        print('using LorentzianModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='Voigt':
                        from lmfit.models import VoigtModel
                        model = VoigtModel()
                        print('using VoigtModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='PseudoVoigt':
                        from lmfit.models import PseudoVoigtModel
                        model = PseudoVoigtModel()
                        print('using PseudoVoigtModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='Moffat':
                        from lmfit.models import MoffatModel
                        model = MoffatModel()
                        print('using MoffatModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='StudentsT':
                        from lmfit.models import StudentsTModel
                        model = StudentsTModel()
                        print('using StudentsTModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='ExponentialGaussian':
                        from lmfit.models import ExponentialGaussianModel
                        model = ExponentialGaussianModel()
                        print('using ExponentialGaussianModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='ExponentialGaussian':
                        from lmfit.models import SplitLorentzianModel
                        model = SplitLorentzianModel()
                        print('using SplitLorentzianModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='Pearson4':
                        from lmfit.models import Pearson4Model
                        model = Pearson4Model()
                        print('using Pearson4Model() instead of GaussianModel() for projection fitting')
                    elif alt_model=='Pearson7':
                        from lmfit.models import Pearson7Model
                        model = Pearson7Model()
                        print('using Pearson7Model() instead of GaussianModel() for projection fitting')
                    elif alt_model=='BreitWigner':
                        from lmfit.models import BreitWignerModel
                        model = BreitWignerModel()
                        print('using BreitWignerModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='SkewedGaussian':
                        from lmfit.models import SkewedGaussianModel
                        model = SkewedGaussianModel()
                        print('using SkewedGaussianModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='Doniach':
                        from lmfit.models import DoniachModel
                        model = DoniachModel()
                        print('using DoniachModel() instead of GaussianModel() for projection fitting')
                    elif alt_model=='Lognormal':
                        from lmfit.models import LognormalModel
                        model = LognormalModel()
                        print('using LognormalModel() instead of GaussianModel() for projection fitting')
                    else:
                        model = GaussianModel()
                        print(alt_model,'model NOT FOUND; using GaussianModel() for projection fitting')
                else:
                    model = GaussianModel()
                x = np.array(x_vals)
                y = np.array(y_vals) * (plot_bin_width_cm / pixel_width) / div_by_norm_factor
                if subtract_constant_background: # scoot distribution to have 0 baseline
                    y_baseline = 0.5*(y[0]+y[-1])
                    y = y - y_baseline
                #ymax_fit = max(y)
                params = model.guess(y, x=x)
                result = model.fit(y, params, x=x)
                # Now do a second attempt at fit based just on the found peak
                H_sig = result.params['sigma'].value
                if not use_proj_peak_window:
                    H_mu = result.params['center'].value
                    mask = (x > H_mu - (2.355 / 2) * H_sig * num_fwhm_in_peak_fit) & (
                                x < H_mu + (2.355 / 2) * H_sig * num_fwhm_in_peak_fit)
                else:
                    H_mu = xavg_max_count  # x[list(y).index(max(y))]
                    mask = (x > H_mu - 10 * proj_lmfit_peak_window_width_cm / 2) & (
                                x < H_mu + 10 * proj_lmfit_peak_window_width_cm / 2)
                #amp_max_fit = ymax_fit*np.sqrt(2*np.pi)*result.params['sigma'].value
                params = model.guess(y[mask], x=x[mask])
                #params['amplitude'] = Parameter(name='amplitude', value=amp_max_fit, min=0.5*amp_max_fit, max=2*amp_max_fit, vary=True)
                #params = model.guess(y[mask], x=x[mask])
                #params = model.make_params(params) #c=y.mean(),center=x.mean(),sigma=x.std(),amplitude=x.std()*y.ptp())
                result = model.fit(y[mask], params, x=x[mask])
                # https://lmfit.github.io/lmfit-py/model.html#modelresult-attributes
                print('lmfit results for {:} image {:} projection peak fit:'.format(ng_plaintxt[:-1],axes_list[non_match_i[ai]]),
                      'sigma = {:g}, '.format(result.params['sigma'].value), ' center = {:g}'.format(result.params['center'].value))
                H_lm_sigma = result.params['sigma'].value
                H_lm_center = result.params['center'].value
                H_lm_center_err = result.params['center'].stderr
                try:
                    H_lm_amplitude = result.params['height'].value
                    H_lm_fwhm = result.params['fwhm'].value
                except:
                    H_lm_amplitude = result.params['amplitude'].value
                    H_lm_fwhm = 2.35*H_lm_sigma
                lmfit_peak_centers.append(H_lm_center)
                lmfit_peak_center_errs.append(H_lm_center_err)

                x_Gauss_fit = np.linspace(-1000, 1000, 200)
                y_Gauss_fit, di_in = eval_distribution(x_Gauss_fit, mu=H_lm_center, a=H_lm_amplitude, sigma=H_lm_sigma)
                if subtract_constant_background:
                    y_Gauss_fit = y_Gauss_fit + y_baseline
                fit_fcn_str = di_in.fcn_tex_str
                fit_fcn_full_name = di_in.full_name
                xdata_list.append(x_Gauss_fit)
                ydata_list.append(y_Gauss_fit)
                # this_color = colors_list[ifi]
                # this_color = (this_color[0], this_color[1], this_color[2], 0.2) # adjust alpha value of fit curve
                try:
                    peak_loc_str = '{:}'.format(axes_list[non_match_i[ai]]) + r'$_{peak}$ = ' + '{:.4g} +/- {:.4g} mm'.format(
                        lmfit_peak_centers[-1], lmfit_peak_center_errs[-1])
                except:
                    peak_loc_str = '{:}'.format(axes_list[non_match_i[ai]]) + r'$_{peak}$ = ' + str(lmfit_peak_centers[-1]) + ' +/- ' + str(lmfit_peak_center_errs[-1]) + ' mm'

            else:
                lmfit_peak_centers.append(xavg_max_count)
                lmfit_peak_center_errs.append(0)
                try:
                    peak_loc_str = '{:}'.format(axes_list[non_match_i[ai]]) + r'$_{peak}$ = ' + '{:.4g} +/- {:.4g} mm'.format(
                        lmfit_peak_centers[-1], lmfit_peak_center_errs[-1])
                except:
                    peak_loc_str = '{:}'.format(axes_list[non_match_i[ai]]) + r'$_{peak}$ = ' + str(lmfit_peak_centers[-1]) + ' +/- ' + str(lmfit_peak_center_errs[-1]) + ' mm'
                # peak_loc_str_list.append(peak_loc_str)
                # peak_loc_str_cols_list.append(plot_colors_list[-1])
                result = None

            plot_dict = Munch({
                'xdata': [plotx],
                'ydata': [ploty],
                'x_label_str': x_str,
                'y_label_str': y_str,
                'mean_str': mean_str,
                'title_str': title_str,
                'mean': mean_val,
                'std_dev': stdv_val,
                'SE_of_mean': std_err_of_mean,
                'lmfit_peak_fit_result':result,
                'peak_loc_str':peak_loc_str,
                'peak_loc_mu':lmfit_peak_centers[-1],
                'peak_loc_err': lmfit_peak_center_errs[-1]
            })
            if ai == 0:
                if ict == 0:
                    results_dict.image.hproj = plot_dict
                elif ict == 1:
                    results_dict.image.hproj.n = plot_dict
                elif ict == 2:
                    results_dict.image.hproj.g = plot_dict
                pixel_width = results_dict.meta.pixel_vwidth
            else:
                if ict == 0:
                    results_dict.image.vproj = plot_dict
                elif ict == 1:
                    results_dict.image.vproj.n = plot_dict
                elif ict == 2:
                    results_dict.image.vproj.g = plot_dict
                pixel_width = results_dict.meta.pixel_hwidth

            fig3, ax3 = fancy_plot(
                xdata_list,
                ydata_list,
                x_scale='linear',
                y_scale='linear',
                x_label_str=x_str,
                y_label_str=y_str,
                title_str=title_str,
                marker='',
                linestyle=['-', '--'],
                color='k',
                # alpha=0.2,
                # x_limits=im_lr_range,#[-500,500],
                # y_limits=im_bt_range,#[-500,500],
                fig_width_inch=6.5,
                fig_height_inch=4.5,
                figi=figi,
            )
            figi += 1
            allaxes = fig3.get_axes()
            allaxes[0].text(0.72, 0.79, mean_str, fontsize=12, transform=ax3.transAxes)
            allaxes[0].text(0.03, 0.92, peak_loc_str, ha='left', color='k', transform=ax3.transAxes, fontsize=12)
            if save_plots:
                for ext in image_extensions:
                    plot_save_path = images_path + ng_plaintxt + '{:}proj'.format(
                        axes_list[non_match_i[ai]]) + '_' + MCtrutxt[:-1] + '_{:}-fit'.format(alt_model) + ext  # or use fig.canvas.get_window_title()
                    # plot_save_path = images_path + slugify(title_str) + MCtrutxt + ext # or use fig.canvas.get_window_title()
                    fig3.savefig(plot_save_path, facecolor=(0, 0, 0, 0))

    with open(image_results_pickle_path, 'wb') as handle:
        to_be_pickled = results_dict
        pickle.dump(Munch(to_be_pickled), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Pickle file written:', image_results_pickle_path, '\n')
    with open(image_results_pickle_path2, 'wb') as handle:
        to_be_pickled = results_dict
        pickle.dump(Munch(to_be_pickled), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Pickle file written:', image_results_pickle_path2, '\n')

    print("\nDone.   ({:0.2f} seconds elapsed)".format(time.time() - start))
    if show_plots:
        plt.show()
