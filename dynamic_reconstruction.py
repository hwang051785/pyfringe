# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:52:02 2024

@author: kl001
"""

import glob
import os
import cv2
from tqdm import tqdm
import sys
sys.path.append(r"C:\Users\kl001\pyfringe")
import nstep_fringe as nstep
import nstep_fringe_cp as nstep_cp
import reconstruction as rc
from plyfile import PlyData, PlyElement
import numpy as np
import cupy as cp

EPSILON = -0.5
TAU = 5.5

def load_path(data_dir, N_list):

    num_images = np.sum(N_list)
    full_path = np.array(glob.glob(os.path.join(data_dir,'*.tiff')))
    path_lst = full_path.reshape(int(len(full_path)/num_images), num_images)
    return path_lst

def cloud_save(reconst_inst,reconst_index):
    saving_path = os.path.join(reconst_inst.object_path,"obj_%d.ply"%reconst_index)
    xyz = list(map(tuple, reconst_inst.coords)) 
    color = list(map(tuple, reconst_inst.inte_rgb))
    if reconst_inst.temp:
        temperature_vector = np.array(reconst_inst.temperature_vector, dtype=[('temperature', 'f4')])
    else:
        temperature_vector = [None]
    
    if reconst_inst.probability:
        xyz_sigma = list(map(tuple, reconst_inst.cordi_sigma))
        xyz_quality = np.array(reconst_inst.quality_vector, dtype=[('quality', 'f4')])
    else:
        xyz_sigma = [None]
        xyz_quality = [None]
        
    PlyData(
        [
            PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
            PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
            PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
            PlyElement.describe(np.array(temperature_vector, dtype=[('temperature', 'f4')]), 'temperature'),
            PlyElement.describe(np.array(xyz_quality, dtype=[('quality', 'f4')]), 'quality'),
        ]).write(saving_path)
    print("\n Point cloud saved at %s"% (saving_path))
    return

def indiv_reconst(data_path, reconst_inst):
    images_arr = np.array([cv2.imread(file,0) for file in data_path])- reconst_inst.dark_bias
    if reconst_inst.processing == 'cpu':
        modulation_vector, orig_img, phase_map, mask = nstep.phase_cal(images_arr,
                                                                       reconst_inst.limit, 
                                                                       reconst_inst.N_list,
                                                                       False)
        reconst_inst.mask = mask
        phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
        unwrap_vector, k_arr, mask = nstep.multifreq_unwrap(reconst_inst.pitch_list,
                                                      phase_map,
                                                      reconst_inst.kernel,
                                                      reconst_inst.fringe_direc,
                                                      reconst_inst.mask,
                                                      reconst_inst.cam_width,
                                                      reconst_inst.cam_height)
        orig_img = orig_img[-1] 
        reconst_inst.mask = mask
        if reconst_inst.probability:
            cov_arr_l,_ = nstep.pred_var_fn(images_arr[-(reconst_inst.N_list[-2]+reconst_inst.N_list[-1]): -reconst_inst.N_list[-1]], reconst_inst.model)
            
            sigma_sq_phi_l = nstep.var_func(images_arr[-(reconst_inst.N_list[-2]+reconst_inst.N_list[-1]): -reconst_inst.N_list[-1]],
                                          reconst_inst.mask,
                                          reconst_inst.N_list[-2],
                                          cov_arr_l)
            cov_arr_h,_ = nstep.pred_var_fn(images_arr[-reconst_inst.N_list[-1]:], reconst_inst.model)
            sigma_sq_phi = nstep.var_func(images_arr[-reconst_inst.N_list[-1]:],
                                          reconst_inst.mask,
                                          reconst_inst.N_list[-1],
                                          cov_arr_h)
            sigma_sq_delta_phi = ((reconst_inst.pitch_list[-2]/reconst_inst.pitch_list[-1])**2 * sigma_sq_phi_l) + sigma_sq_phi
            quality = np.pi/np.sqrt(sigma_sq_delta_phi)
            
        else:
            sigma_sq_phi = None
            quality = None
    elif reconst_inst.processing == 'gpu':
        images_arr_cp = cp.asarray(images_arr)
        modulation_vector, orig_img, phase_map, mask = nstep_cp.phase_cal_cp(images_arr_cp,
                                                                             reconst_inst.limit,
                                                                             reconst_inst.N_list,
                                                                             False)
        phase_map[0][phase_map[0] < EPSILON] = phase_map[0][phase_map[0] < EPSILON] + 2 * np.pi
        reconst_inst.mask = mask
        unwrap_vector, k_arr, mask = nstep_cp.multifreq_unwrap_cp(reconst_inst.pitch_list,
                                                            phase_map,
                                                            reconst_inst.kernel,
                                                            reconst_inst.fringe_direc,
                                                            reconst_inst.mask,
                                                            reconst_inst.cam_width,
                                                            reconst_inst.cam_height)
        orig_img = cp.asnumpy(orig_img[-1])
        reconst_inst.mask = mask
        if reconst_inst.probability:
            
            cov_arr_l,_ = nstep_cp.pred_var_fn(images_arr_cp[-(reconst_inst.N_list[-2]+reconst_inst.N_list[-1]): -reconst_inst.N_list[-1]], reconst_inst.model)
            
            sigma_sq_phi_l = nstep_cp.var_func(images_arr_cp[-(reconst_inst.N_list[-2]+reconst_inst.N_list[-1]): -reconst_inst.N_list[-1]],
                                          reconst_inst.mask,
                                          reconst_inst.N_list[-2],
                                          cov_arr_l)
            cov_arr_h,_ = nstep_cp.pred_var_fn(images_arr_cp[-reconst_inst.N_list[-1]:], reconst_inst.model)
            sigma_sq_phi = nstep_cp.var_func(images_arr_cp[-reconst_inst.N_list[-1]:],
                                          reconst_inst.mask,
                                          reconst_inst.N_list[-1],
                                          cov_arr_h)
            sigma_sq_delta_phi = ((reconst_inst.pitch_list[-2]/reconst_inst.pitch_list[-1])**2 * sigma_sq_phi_l) + sigma_sq_phi
            quality = np.pi/np.sqrt(sigma_sq_delta_phi)
            quality = cp.asnumpy(quality)
        else:
            sigma_sq_phi = None
            quality = None
    if os.path.exists(os.path.join(reconst_inst.object_path, 'white.tiff')):
        inte_img = cv2.imread(os.path.join(reconst_inst.object_path, 'white.tiff'))
        inte_rgb_image = inte_img[..., ::-1].copy()
    else:
        inte_rgb_image = orig_img
    temperature_image = None
    obj_cordi, obj_color, cordi_sigma = reconst_inst.complete_recon(unwrap_vector,                                                
                                                             inte_rgb_image,
                                                             temperature_image,
                                                             sigma_sq_phi,
                                                             quality)
    
    return obj_cordi, obj_color, cordi_sigma

def dynamic_reconst(reconst_inst):
    path_arr = load_path(reconst_inst.object_path, reconst_inst.N_list)
    
    for i,p in tqdm(enumerate(path_arr),desc="Reconstruction"):
        obj_cordi, obj_color, cordi_sigma = indiv_reconst(p, reconst_inst)
        cloud_save(reconst_inst,i)
    return

def main():
    print("\nPlease Choose")
    option = input("\n2: 2 level reconstruction \n3: 3 level reconstruction \n4: 4 level reconstruction")
    if option == "2":
       pitch_list =[1200, 18]
      # N_list = [3, 3]
       N_list = [3, 3]
    elif option == "3":
       pitch_list = [1200, 120, 12]
       N_list = [3, 3, 9]
    elif option == "4":
       pitch_list =[1375, 275, 55, 11] 
       N_list = [3, 3, 3, 9]
    else:
       print("ERROR: Invalid entry for number of levels")
       return
    limit = float(input("\nEnter background limit:"))
    proj_width = 912  
    proj_height = 1140 
    cam_width = 1920 
    cam_height = 1200
    type_unwrap = 'multifreq'
    data_dir = r"E:\test2"
    calib_path = r"G:\My Drive\Epistemic_newdata\calibration_100"
    model_path = r"G:\My Drive\Epistemic_newdata\variance_model.npy"
    dark_bias_path = r"C:\Users\kl001\Documents\pyfringe_test\mean_pixel_std\exp_30_fp_42_retake\black_bias\avg_dark.npy"
    reconst_inst = rc.Reconstruction(proj_width=proj_width,
                                     proj_height=proj_height,
                                     cam_width=cam_width,
                                     cam_height=cam_height,
                                     type_unwrap=type_unwrap,
                                     limit=limit,
                                     N_list=N_list,
                                     pitch_list=pitch_list,
                                     fringe_direc='v',
                                     kernel=7,
                                     data_type='tiff',
                                     processing='gpu',
                                     dark_bias_path=dark_bias_path,
                                     calib_path=calib_path,
                                     object_path=data_dir,
                                     model_path=model_path,
                                     temp=None,
                                     save_ply=False,
                                     probability=True,
                                     prob_up=True)
    
    dynamic_reconst(reconst_inst)
    return

if __name__ == '__main__':
    main()