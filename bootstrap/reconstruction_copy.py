#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:00:19 2022

@author: Sreelakshmi
"""
import numpy as np
import nstep_fringe as nstep
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os
from copy import deepcopy
from plyfile import PlyData, PlyElement

EPSILON = -0.5
TAU = 5.5
## ideal if this fubction added to nstep
def B_cutoff_limit(sigma_path, quantile_limit, N_list, pitch_list):
    '''
    Function to calculate modulation minitue based on sucess rate

    Parameters
    ----------
    sigma_path = type:string. Path to read variance of noise model (sigma) 
    quantile_limit = type: float. Sigma level upto which all pixels can be successfully unwrapped.
    N_list = type: Array of int. Number of images taken for each level.
    pitch_list =type: Array of int. Number of pixels per fringe period in each level
    Returns
    -------
    Lower limit of modulation. Pixels above this value is used for reconstruction.

    '''
    sigma = np.load(sigma_path)
    sigma_sq_delta_phi = (np.pi / quantile_limit)**2
    modulation_limit_sq = ((pitch_list[-1] / pitch_list[-2]) + 1) * (2 * sigma**2) / (N_list[-1]* sigma_sq_delta_phi)
    
    return np.sqrt(modulation_limit_sq)


def inv_mtx(a11,a12,a13,a21,a22,a23,a31,a32,a33):
    '''
    Function to calculate inversion matrix required for object reconstruction.
    Ref: S.Zhong, High-Speed 3D Imaging with Digital Fringe Projection Techniques, CRC Press, 2016.
    '''
   
    
    det = (a11 * a22 * a33) + (a12 * a23 * a31) + (a13 * a21 * a32) - (a13 * a22 * a31) - (a12 * a21 * a33) - (a11* a23* a32)    
    
    b11 = (a22 * a33 - a23 * a32) / det 
    b12 = -(a12 * a33 - a13 * a32) / det 
    b13 = (a12 * a23 - a13 * a22) / det
    
    b21 = -(a21 * a33 - a23 * a31) / det
    b22 = (a11 * a33 - a13 * a31) / det
    b23 = -(a11 * a23 - a13 * a21) / det
    
    b31 = (a21 * a32 - a22 * a31) / det
    b32 = -(a11 * a32 - a12 * a31) / det
    b33 = (a11 * a22 - a12 * a21) / det
    
    #b_mtx=np.stack((np.vstack((b11,b12,b13)).T,np.vstack((b21,b22,b23)).T,np.vstack((b31,b32,b33)).T),axis=1)
    
    return b11, b12, b13, b21, b22, b23, b31, b32, b33
    
    
    
    
def reconstruction_pts(uv_true, unwrapv, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch):
    '''
    Function to reconstruct 3D point cordinates of 2D points. 

    Parameters
    ----------
    uv_true = type: float. 2D point cordinates
    unwrapv = type: float array. Unwrapped phase map of object.
    c_mtx = type: float array. Camera matrix from calibration.
    c_dist = type: float array. Camera distortion matrix from calibration.
    p_mtx = type: float array. Projector matrix from calibration.
    cp_rot_mtx = type: float array. Projector distortion matrix from calibration.
    cp_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type:float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.

    Returns
    -------
    Coordinates array for given 2D points
    x = type: float. 
    y = type: float. 
    z = type: float. 

    '''
    no_pts = uv_true.shape[0]
    uv = cv2.undistortPoints(uv_true, c_mtx, c_dist, None, c_mtx )
    uv = uv.reshape(uv.shape[0],2)
    uv_true = uv_true.reshape(no_pts,2)
    #  Extract x and y coordinate of each point as uc, vc
    uc = uv[:,0].reshape(no_pts,1)
    vc = uv[:,1].reshape(no_pts,1)
    
    # Determinate 'up' from circle center
    up = np.array([(nstep.bilinear_interpolate(unwrapv,i) - phase_st) * (pitch / (2*np.pi)) for i in uv_true])
    up = up.reshape(no_pts,1)
    
    # Calculate H matrix for proj from intrinsics and extrinsics
    proj_h_mtx = np.dot(p_mtx, np.hstack((cp_rot_mtx, cp_trans_mtx)))
    #Calculate H matrix for camer
    cam_h_mtx = np.dot(c_mtx,np.hstack((np.identity(3), np.zeros((3,1)))))
    
    a11 = cam_h_mtx[0,0] - uc * cam_h_mtx[2,0]
    a12 = cam_h_mtx[0,1] - uc * cam_h_mtx[2,1]
    a13 = cam_h_mtx[0,2] - uc * cam_h_mtx[2,2]
    
    a21 = cam_h_mtx[1,0] - vc * cam_h_mtx[2,0]
    a22 = cam_h_mtx[1,1] - vc * cam_h_mtx[2,1]
    a23 = cam_h_mtx[1,2] - vc * cam_h_mtx[2,2]
    
    a31 = proj_h_mtx[0,0] - up * proj_h_mtx[2,0]
    a32 = proj_h_mtx[0,1] - up * proj_h_mtx[2,1]
    a33 = proj_h_mtx[0,2] - up * proj_h_mtx[2,2]
    
    b11, b12, b13, b21, b22, b23, b31, b32, b33 = inv_mtx(a11, a12, a13, a21, a22, a23, a31, a32,a33)
    
    c1 = uc * cam_h_mtx[2,3] - cam_h_mtx[0,3]
    c2 = vc * cam_h_mtx[2,3] - cam_h_mtx[1,3]
    c3 = up * proj_h_mtx[2,3] - proj_h_mtx[0,3]
   
    x = b11 * c1 + b12 * c2 + b13 * c3
    y = b21 * c1 + b22 * c2 + b23 * c3
    z = b31 * c1 + b32 * c2 + b33 * c3
    return x, y, z

def point_error(cord1,cord2):
    '''
    Function to plot error 

    '''
    
    delta = cord1 - cord2
    abs_delta = abs(delta)
    err_df =  pd.DataFrame(np.hstack((delta,abs_delta)) , columns = ['$\Delta x$','$\Delta y$','$\Delta z$','$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$'])
    plt.figure()
    gfg = sns.histplot(data = err_df[['$abs(\Delta x)$', '$abs(\Delta y)$', '$abs(\Delta z)$']])
    plt.xlabel('Absolute error mm',fontsize = 30)
    plt.ylabel('Count',fontsize = 30)
    plt.title('Reconstruction error',fontsize=30)
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlim(0,3)
    plt.setp(gfg.get_legend().get_texts(), fontsize='20') 
    return err_df
    
def reconstruction_obj(unwrapv, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch):
    '''
    Sub function to reconstruct object from phase map

    Parameters
    ----------
    unwrapv = type: float array. Unwrapped phase map of object.
    c_mtx = type: float array. Camera matrix from calibration.
    c_dist = type: float array. Camera distortion matrix from calibration.
    p_mtx = type: float array. Projector matrix from calibration.
    cp_rot_mtx = type: float array. Projector distortion matrix from calibration.
    cp_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.

    Returns
    -------
    Coordinates array for all points
    x = type: float array . 
    y = type: float array. 
    z = type: float array. 
    '''
    
    unwrap_dist = cv2.undistort(unwrapv, c_mtx, c_dist)
    u = np.arange(0,unwrap_dist.shape[1])
    v = np.arange(0,unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u,v)
    up = (unwrap_dist - phase_st) * pitch / (2*np.pi) 
    # Calculate H matrix for proj from intrinsics and extrinsics
    proj_h_mtx = np.dot(p_mtx, np.hstack((cp_rot_mtx, cp_trans_mtx)))

    #Calculate H matrix for camera
    cam_h_mtx = np.dot(c_mtx,np.hstack((np.identity(3), np.zeros((3,1)))))

    a11 = cam_h_mtx[0,0] - uc * cam_h_mtx[2,0] 
    a12 = cam_h_mtx[0,1] - uc * cam_h_mtx[2,1]
    a13 = cam_h_mtx[0,2] - uc * cam_h_mtx[2,2]

    a21 = cam_h_mtx[1,0] - vc * cam_h_mtx[2,0]
    a22 = cam_h_mtx[1,1] - vc * cam_h_mtx[2,1]
    a23 = cam_h_mtx[1,2] - vc * cam_h_mtx[2,2]

    a31 = proj_h_mtx[0,0] - up * proj_h_mtx[2,0]
    a32 = proj_h_mtx[0,1] - up * proj_h_mtx[2,1]
    a33 = proj_h_mtx[0,2] - up * proj_h_mtx[2,2]

    b11, b12, b13, b21, b22, b23, b31, b32, b33 = inv_mtx(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    
    c1 = uc * cam_h_mtx[2,3] - cam_h_mtx[0,3]
    c2 = vc * cam_h_mtx[2,3] - cam_h_mtx[1,3]
    c3 = up * proj_h_mtx[2,3] - proj_h_mtx[0,3]
    
    x = b11 * c1 + b12 * c2 + b13 * c3
    y = b21 * c1 + b22 * c2 + b23 * c3
    z = b31 * c1 + b32 * c2 + b33 * c3
    
    return x, y, z 

def diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up):
    '''
    Subfunction used to calculate x cordinate variance

    '''
    df_dup = (det * (-hc_13 * hc_22 * hp_34 + uc * hc_22 * hc_33 * hp_34) - x_num * (-hc_11 * hc_22 * hp_33 + hc_13 * hc_22 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
    df_dhc_11 = ( - x_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
    df_dhc_13 = (det * (-up * hc_22 * hp_34 + hc_22 * hp_14) - x_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
    df_dhc_22 = (det * (-up * hc_13 * hp_34 + hc_13 * hp_14 + uc * up * hc_33 * hp_34 - uc * hc_33 * hp_14) - x_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
    df_dhc_23 = ( - x_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
    df_dhc_33 = (det * (uc* up * hc_22 * hp_34 - uc * hc_22 * hp_14) - x_num * (uc * hc_22 * hp_11 - uc* up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32))/det**2
    df_dhp_11 = ( - x_num *( -hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
    df_dhp_12 = ( - x_num * (-hc_11*hc_23 + vc * hc_11 * hc_33))/det**2
    df_dhp_13 = ( - x_num * (hc_11 * hc_22))/det**2
    df_dhp_14 = (det * (hc_13 * hc_22 - uc * hc_22 * hc_33))/det**2
    df_dhp_31 = ( - x_num * (up * hc_22 * hc_13 - uc * up * hc_22 * hc_33))/det**2
    df_dhp_32 = ( - x_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
    df_dhp_33 = ( - x_num * (-up * hc_11 * hc_22))/det**2
    df_dhp_34 = (det * (-up * hc_13 * hc_22 + uc * up * hc_22 * hc_33))/det**2
    
    return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33,df_dhp_34

def diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up):
    '''
    Subfunction used to calculate y cordinate variance

    '''
    df_dup = (det * (-hc_11 * hc_23 * hp_34 + vc * hc_11 * hc_33 * hp_34) - y_num * (-hc_11 * hc_22 * hp_33 + hc_13 * hc_22 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
    df_dhc_11 = (det * (-up * hc_23 * hp_34 + hc_23 * hp_14 + vc * up * hc_33 * hp_34 - vc * hc_33 * hp_14) - y_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
    df_dhc_13 = ( - y_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
    df_dhc_22 = ( - y_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
    df_dhc_23 = (det * (-up * hc_11 * hp_34 + hc_11 * hp_14) - y_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
    df_dhc_33 = (det * (vc* up * hc_11 * hp_34 - vc * hc_11 * hp_14) - y_num * (uc * hc_22 * hp_11 - uc * up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32))/det**2
    df_dhp_11 = ( - y_num * (-hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
    df_dhp_12 = ( - y_num * (-hc_11 * hc_23 + vc * hc_11 * hc_33))/det**2
    df_dhp_13 = ( - y_num * (hc_11 * hc_22))/det**2
    df_dhp_14 = (det * (hc_11 * hc_23 - vc * hc_11 * hc_33))/det**2
    df_dhp_31 = ( - y_num * ( up * hc_13 * hc_22 - uc * up * hc_22 * hc_33))/det**2
    df_dhp_32 = ( - y_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
    df_dhp_33 = ( - y_num * (-up * hc_11 * hc_22))/det**2
    df_dhp_34 = (det * (-up * hc_11 * hc_23 + vc * up *  hc_11 * hc_33))/det**2
    
    return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33,df_dhp_34

def diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up):
    '''
    Subfunction used to calculate z cordinate variance

    '''
    df_dup = (det * (hc_11 * hc_22 * hp_34) - z_num * (-hc_11 * hc_22 * hp_33 + hc_22 * hc_13 * hp_31 - uc * hc_22 * hc_33 * hp_31 + hc_11 * hc_23 * hp_32 - vc * hc_11 * hc_33 * hp_32))/det**2
    df_dhc_11 = (det * (up * hc_22 * hp_34 - hc_22 * hp_14) - z_num * (hc_22 * hp_13 - up * hc_22 * hp_33 - hc_23 * hp_12 + up * hc_23 * hp_32 + vc * hc_33 * hp_12 - vc * up * hc_33 * hp_32))/det**2
    df_dhc_13 = ( - z_num * (-hc_22 * hp_11 + up * hc_22 * hp_31))/det**2
    df_dhc_22 = (det * (up * hc_11 * hp_34 - hc_11 * hp_14) - z_num * (hc_11 * hp_13 - up * hc_11 * hp_33 - hc_13 * hp_11 + up * hc_13 * hp_31 + uc * hc_33 * hp_11 - uc * up * hc_33 * hp_31))/det**2
    df_dhc_23 = ( - z_num * (-hc_11 * hp_12 + up * hc_11 * hp_32))/det**2
    df_dhc_33 = (- z_num * (uc * hc_22 * hp_11 - uc * up * hc_22 * hp_31 + vc * hc_11 * hp_12 - vc * up * hc_11 * hp_32))/det**2
    df_dhp_11 = ( - z_num * (-hc_13 * hc_22 + uc * hc_22 * hc_33))/det**2
    df_dhp_12 = ( - z_num * (-hc_11 * hc_23 + vc * hc_11 * hc_33))/det**2
    df_dhp_13 = ( - z_num * (hc_11 * hc_22))/det**2
    df_dhp_14 = (det * (-hc_11 * hc_22))/det**2
    df_dhp_31 = ( - z_num * (up * hc_22 * hc_13 - uc * up * hc_22 * hc_33))/det**2
    df_dhp_32 = ( - z_num * (up * hc_11 * hc_23 - vc * up * hc_11 * hc_33))/det**2
    df_dhp_33 = ( - z_num * (-up * hc_11 * hc_22 ))/det**2
    df_dhp_34 = (det * (up * hc_11 * hc_22))/det**2
    
    return df_dup, df_dhc_11, df_dhc_13, df_dhc_22, df_dhc_23, df_dhc_33, df_dhp_11, df_dhp_12, df_dhp_13, df_dhp_14, df_dhp_31, df_dhp_32, df_dhp_33,df_dhp_34

def sigma_random(modulation, limit,  pitch, N, phase_st, unwrap, sigma_path, source_folder):
    '''
    Function to calculate variance of x,y,z cordinates

    '''
    sigma = np.load(sigma_path)
    mean_calibration_param = np.load(os.path.join(source_folder,'mean_calibration_param.npz'))
    h_matrix_param = np.load(os.path.join(source_folder,'h_matrix_param.npz'))
    c_mtx = mean_calibration_param["arr_0"]
    c_dist = mean_calibration_param["arr_2"]
    proj_h_mtx_mean = h_matrix_param["arr_2"]
    cam_h_mtx_mean = h_matrix_param["arr_0"]
    proj_h_mtx_std = h_matrix_param["arr_3"]
    cam_h_mtx_std = h_matrix_param["arr_1"]
    
    
    mod_copy = deepcopy(modulation)
    unwrap_copy = deepcopy(unwrap)
    roi_mask = np.full(mod_copy.shape, False)
    roi_mask[mod_copy > limit] = True
    mod_copy[~roi_mask] = np.nan
    unwrap_copy[~roi_mask] = np.nan
    unwrap_dist = cv2.undistort(unwrap, c_mtx, c_dist)
    u = np.arange(0,unwrap_dist.shape[1])
    v = np.arange(0,unwrap_dist.shape[0])
    uc, vc = np.meshgrid(u,v)
    up = (unwrap_dist - phase_st) * pitch / (2*np.pi)
    sigma_sq_phi = (2 * sigma**2) / (N * mod_copy**2)
    sigma_sq_up = sigma_sq_phi * pitch**2 / 4 * np.pi**2
    
    hc_11 = cam_h_mtx_mean[0,0]
    sigmasq_hc_11 = cam_h_mtx_std[0,0]**2
    hc_13 = cam_h_mtx_mean[0,2]
    sigmasq_hc_13 = cam_h_mtx_std[0,2]**2
    
    hc_22 = cam_h_mtx_mean[1,1]
    sigmasq_hc_22 = cam_h_mtx_std[1,1]**2
    hc_23 = cam_h_mtx_mean[1,2]
    sigmasq_hc_23 = cam_h_mtx_std[1,2]**2
    
    hc_33 = cam_h_mtx_mean[2,2]
    sigmasq_hc_33 = cam_h_mtx_std[2,2]**2
    
    hp_11 = proj_h_mtx_mean[0,0]
    sigmasq_hp_11 = proj_h_mtx_std[0,0]**2
    hp_12 = proj_h_mtx_mean[0,1]
    sigmasq_hp_12 = proj_h_mtx_std[0,1]**2
    hp_13 = proj_h_mtx_mean[0,2]
    sigmasq_hp_13 = proj_h_mtx_std[0,2]**2
    hp_14 = proj_h_mtx_mean[0,3]
    sigmasq_hp_14 = proj_h_mtx_std[0,3]**2
    
    hp_31 = proj_h_mtx_mean[2,0]
    sigmasq_hp_31 = proj_h_mtx_std[2,0]**2
    hp_32 = proj_h_mtx_mean[2,1]
    sigmasq_hp_32 = proj_h_mtx_std[2,1]**2
    hp_33 = proj_h_mtx_mean[2,2]
    sigmasq_hp_33 = proj_h_mtx_std[2,2]**2
    hp_34 = proj_h_mtx_mean[2,3]
    sigmasq_hp_34 = proj_h_mtx_std[2,3]**2
    
    det = (hc_11 * hc_22 * hp_13 - up * hc_11 * hc_22 * hp_33 - hc_13 * hc_22 * hp_11 + up * hc_13 *hc_22 * hp_31 + uc * hc_22 * hc_33 * hp_11
           - uc * up * hc_22 * hc_33 * hp_31 - hc_11 * hc_23 * hp_12 + up * hc_11 * hc_23 * hp_32 + vc * hc_11 * hc_33 * hp_12 - vc * up * hc_11 * hc_33 * hp_32)
           
    
    x_num = -up * hc_13 * hc_22 * hp_34 + hc_13 * hc_22 * hp_14 + uc * up * hc_22 * hc_33 * hp_34 - uc * hc_22 * hc_33 * hp_14
    df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x = diff_funs_x(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, x_num, uc, vc, up)
    sigmasq_x = ((df_dup_x**2 * sigma_sq_up) + (df_dhc_11_x**2 * sigmasq_hc_11) + ( df_dhc_13_x**2 * sigmasq_hc_13) + (df_dhc_22_x**2 * sigmasq_hc_22) + (df_dhc_23_x**2 *  sigmasq_hc_23) + (df_dhc_33_x**2 * sigmasq_hc_33) 
                + (df_dhp_11_x**2 * sigmasq_hp_11) + (df_dhp_12_x**2 * sigmasq_hp_12) + (df_dhp_13_x**2 * sigmasq_hp_13) + (df_dhp_14_x**2 * sigmasq_hp_14) + (df_dhp_31_x**2 * sigmasq_hp_31) + (df_dhp_32_x**2 * sigmasq_hp_32) + ( df_dhp_33_x**2 * sigmasq_hp_33) + (df_dhp_34_x**2 * sigmasq_hp_34))
    sigmasq_x[~roi_mask] = np.nan
    derv_x = np.stack((df_dup_x, df_dhc_11_x, df_dhc_13_x, df_dhc_22_x, df_dhc_23_x, df_dhc_33_x, df_dhp_11_x, df_dhp_12_x, df_dhp_13_x, df_dhp_14_x, df_dhp_31_x, df_dhp_32_x, df_dhp_33_x, df_dhp_34_x))
    
    y_num = -up * hc_11 * hc_23 * hp_34 + hc_11 * hc_23 * hp_14 + vc * up * hc_11 * hc_33 * hp_34 - vc * hc_11 * hc_33 * hp_14
    #y_num = (-hc_11 * (hc_23 - hc_33)*(up * hp_34 - hp_14))
    df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y = diff_funs_y(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, y_num, uc, vc, up)
    sigmasq_y = ((df_dup_y**2 * sigma_sq_up) + (df_dhc_11_y**2 * sigmasq_hc_11) + ( df_dhc_13_y**2 * sigmasq_hc_13) + (df_dhc_22_y**2 * sigmasq_hc_22) + (df_dhc_23_y**2 *  sigmasq_hc_23) + (df_dhc_33_y**2 * sigmasq_hc_33) 
                + (df_dhp_11_y**2 * sigmasq_hp_11) + (df_dhp_12_y**2 * sigmasq_hp_12) + (df_dhp_13_y**2 * sigmasq_hp_13) + (df_dhp_14_y**2 * sigmasq_hp_14) + (df_dhp_31_y**2 * sigmasq_hp_31) + (df_dhp_32_y**2 * sigmasq_hp_32) + ( df_dhp_33_y**2 * sigmasq_hp_33) + (df_dhp_34_y**2 * sigmasq_hp_34))
    sigmasq_y[~roi_mask] = np.nan
    derv_y = np.stack((df_dup_y, df_dhc_11_y, df_dhc_13_y, df_dhc_22_y, df_dhc_23_y, df_dhc_33_y, df_dhp_11_y, df_dhp_12_y, df_dhp_13_y, df_dhp_14_y, df_dhp_31_y, df_dhp_32_y, df_dhp_33_y, df_dhp_34_y))
    
    z_num = up * hc_11 * hc_22 * hp_34 - hc_11 * hc_22 * hp_14 
    df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z = diff_funs_z(hc_11, hc_13, hc_22, hc_23, hc_33, hp_11,hp_12, hp_13, hp_14, hp_31, hp_32, hp_33, hp_34, det, z_num, uc, vc, up)
    sigmasq_z = ((df_dup_z**2 * sigma_sq_up) + (df_dhc_11_z**2 * sigmasq_hc_11) + ( df_dhc_13_z**2 * sigmasq_hc_13) + (df_dhc_22_z**2 * sigmasq_hc_22) + (df_dhc_23_z**2 *  sigmasq_hc_23) + (df_dhc_33_z**2 * sigmasq_hc_33) 
                + (df_dhp_11_z**2 * sigmasq_hp_11) + (df_dhp_12_z**2 * sigmasq_hp_12) + (df_dhp_13_z**2 * sigmasq_hp_13) + (df_dhp_14_z**2 * sigmasq_hp_14) + (df_dhp_31_z**2 * sigmasq_hp_31) + (df_dhp_32_z**2 * sigmasq_hp_32) + ( df_dhp_33_z**2 * sigmasq_hp_33) + (df_dhp_34_z**2 * sigmasq_hp_34))
    sigmasq_z[~roi_mask] = np.nan
    derv_z = np.stack((df_dup_z, df_dhc_11_z, df_dhc_13_z, df_dhc_22_z, df_dhc_23_z, df_dhc_33_z, df_dhp_11_z, df_dhp_12_z, df_dhp_13_z, df_dhp_14_z, df_dhp_31_z, df_dhp_32_z, df_dhp_33_z, df_dhp_34_z))
    
    return sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z

def complete_recon(unwrap, inte_rgb, modulation, limit, calib_path, sigma_path, phase_st, pitch, N, obj_path, temp, temperature = None):
    '''
    Function to completely reconstruct object applying modulation mask to saving point cloud.

    Parameters
    ----------
    unwrap = type: float array. Unwrapped phase map of object.
    inte_rgb = type: float array. Texture image.
    modulation = type: float array. Intensity modulation image.
    limit = type: float. Intensity modulation limit for mask.
    c_mtx = type: float array. Camera matrix from calibration.
    c_dist = type: float array. Camera distortion matrix from calibration.
    p_mtx = type: float array. Projector matrix from calibration.
    cp_rot_mtx = type: float array. Projector distortion matrix from calibration.
    cp_trans_mtx = type: float array. Camera-projector translational matrix from calibration.
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    pitch  = type:float. Number of pixels per fringe period.
    N = type: int. No. of images
    obj_path = type: string. Path to save point 3D reconstructed point cloud. 
    temp = type: bool. True if temperature information is available else False
    temperature = type:Array of floats. Temperature values for each pixel

    Returns
    -------
    cordi = type:  float array. x,y,z coordinate array of each object point.
    intensity = type: float array. Intensity (texture/ color) at each point.

    '''
    calibration = np.load(os.path.join(calib_path,'mean_calibration_param.npz'))
    c_mtx = calibration["arr_0"]
    c_dist = calibration["arr_2"]
    p_mtx = calibration["arr_4"]
    cp_rot_mtx = calibration["arr_8"]
    cp_trans_mtx = calibration["arr_10"]
    
    obj_x, obj_y,obj_z = reconstruction_obj(unwrap, c_mtx, c_dist, p_mtx, cp_rot_mtx, cp_trans_mtx, phase_st, pitch)
    roi_mask = np.full(unwrap.shape, False)
    roi_mask[modulation > limit] = True
    mod = deepcopy(modulation)
    mod[~roi_mask] = np.nan
    mod_vect = np.array(mod.ravel(), dtype=[('modulation', 'f4')])
    #u_copy = deepcopy(unwrap)
    w_copy = deepcopy(inte_rgb)
    #u_copy[~roi_mask] = np.nan
    w_copy[~roi_mask] = False
    obj_x[~roi_mask] = np.nan
    obj_y[~roi_mask] = np.nan
    obj_z[~roi_mask] = np.nan
    cordi = np.vstack((obj_x.ravel(), obj_y.ravel(), obj_z.ravel())).T
    xyz = list(map(tuple, cordi)) 
    inte_rgb = inte_rgb / np.nanmax(inte_rgb)
    rgb_intensity_vect = np.vstack((inte_rgb[:,:,0].ravel(), inte_rgb[:,:,1].ravel(),inte_rgb[:,:,2].ravel())).T
    color = list(map(tuple, rgb_intensity_vect))
    
    sigmasq_x, sigmasq_y, sigmasq_z, derv_x, derv_y, derv_z =  sigma_random(modulation, limit,  pitch, N, phase_st, unwrap, sigma_path, calib_path)
    sigma_x = np.sqrt(sigmasq_x)
    sigma_y = np.sqrt(sigmasq_y)
    sigma_z = np.sqrt(sigmasq_z)
    cordi_sigma = np.vstack((sigma_x.ravel(), sigma_y.ravel(), sigma_z.ravel())).T
    xyz_sigma = list(map(tuple, cordi_sigma))
    if temp:
        #t_vect = np.array(temperature[flag], dtype=[('temperature', 'f4')])
        t_vect = np.array(temperature.ravel(), dtype=[('temperature', 'f4')])
        PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                PlyElement.describe(np.array(t_vect, dtype=[('temperature', 'f4')]), 'temperature'),
                PlyElement.describe(np.array(mod_vect, dtype=[('modulation', 'f4')]), 'modulation')
                
            ]).write(os.path.join(obj_path,'obj.ply'))
    
    else:
        t_vect = None
        PlyData(
            [
                PlyElement.describe(np.array(xyz, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'points'),
                PlyElement.describe(np.array(color, dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4')]), 'color'),
                PlyElement.describe(np.array(xyz_sigma, dtype=[('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4')]), 'std'),
                PlyElement.describe(np.array(mod_vect, dtype=[('modulation', 'f4')]), 'modulation')
                
            ]).write(os.path.join(obj_path,'obj.ply'))
      
    return cordi, rgb_intensity_vect, t_vect, cordi_sigma

def obj_reconst_wrapper(width, height, pitch_list, N_list, limit,  phase_st, direc, type_unwrap, calib_path, obj_path, sigma_path, temp, kernel = 1):
   '''
    Function for 3D reconstruction of object based on different unwrapping method.

    Parameters
    ----------
    width =type: float. Width of projector.
    height = type: float. Height of projector.
    pitch_list : TYPE
        DESCRIPTION.
    N_list = type: float array. The number of steps in phase shifting algorithm. If phase coded unwrapping method is used this is a single element array. For other methods corresponding to each pitch one element in the list.
    limit = type: float array. Array of number of pixels per fringe period.
    
    phase_st = type: float. Initial phase to be subtracted for phase to coordinate conversion.
    direc = type: string. Visually vertical (v) or horizontal(h) pattern.
    type_unwrap = type: string. Type of temporal unwrapping to be applied. 
                  'phase' = phase coded unwrapping method, 
                  'multifreq' = multifrequency unwrapping method
                  'multiwave' = multiwavelength unwrapping method.
    calib_path = type: string. Path to read mean calibration paraneters. 
    obj_path = type: string. Path to read captured images
    kernel = type: int. Kernel size for median filter. The default is 1.

    Returns
    -------
    obj_cordi = type : float array. Array of reconstructed x,y,z coordinates of each points on the object
    obj_color = type: float array. Color (texture/ intensity) at each point.

    '''
   
  # calibration = np.load(os.path.join(calib_path,'{}_calibration_param.npz'.format(type_unwrap)))
  
   if type_unwrap == 'phase':
       object_cos, obj_cos_mod, obj_cos_avg, obj_cos_gamma, delta_deck_cos  = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(0, N_list[0])]), limit)
       object_step, obj_step_mod, obj_step_avg, obj_step_gamma, delta_deck_step = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0],2 * N_list[0])]), limit)

       #wrapped phase
       phase_cos = nstep.phase_cal(object_cos, N_list, delta_deck_cos )
       phase_step = nstep.phase_cal(object_step, N_list, delta_deck_step )
       phase_step = nstep.step_rectification(phase_step,direc)
       #unwrapped phase
       unwrap0, k0 = nstep.unwrap_cal(phase_step, phase_cos, pitch_list[0], width, height, direc)
       unwrap, k = nstep.filt(unwrap0, kernel, direc)
       inte_img = cv2.imread(os.path.join(obj_path,'white.jpg'))   
       if temp:
           temperature = np.load(os.path.join(obj_path,'temperature.npy'))
       else:
           temperature = 0
       inte_rgb = inte_img[...,::-1].copy()
       np.save(os.path.join(obj_path,'{}_obj_mod.npy'.format(type_unwrap)),obj_cos_mod) 
       np.savez(os.path.join(obj_path,'{}_unwrap.npz'.format(type_unwrap)),data = unwrap.data, mask = unwrap.mask)
       obj_cordi, obj_color, obj_t, cordi_sigma = complete_recon(unwrap, inte_rgb, obj_cos_mod, limit, calib_path, sigma_path, phase_st, pitch_list[-1], N_list[-1], obj_path, temp, temperature)
       
   elif type_unwrap == 'multifreq':
       object_freq1, mod_freq1, avg_freq1, gamma_freq1, delta_deck_freq1  = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(0, N_list[0])]), limit)
       object_freq2, mod_freq2, avg_freq2, gamma_freq2, delta_deck_freq2 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0], N_list[0] + N_list[1])]), limit)
       object_freq3, mod_freq3, avg_freq3, gamma_freq3, delta_deck_freq3 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range( N_list[0] + N_list[1], N_list[0]+ N_list[1]+ N_list[2])]), limit)
       object_freq4, mod_freq4, avg_freq4, gamma_freq4, delta_deck_freq4 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0]+ N_list[1]+ N_list[2], N_list[0]+ N_list[1]+ N_list[2] + N_list[3])]), limit)

       #wrapped phase
       phase_freq1 = nstep.phase_cal(object_freq1, N_list[0], delta_deck_freq1 )
       phase_freq2 = nstep.phase_cal(object_freq2, N_list[1], delta_deck_freq2 )
       phase_freq3 = nstep.phase_cal(object_freq3, N_list[2], delta_deck_freq3 )
       phase_freq4 = nstep.phase_cal(object_freq4, N_list[3], delta_deck_freq4 )
       phase_freq1[phase_freq1 < EPSILON] = phase_freq1[phase_freq1 < EPSILON] + 2 * np.pi

       #unwrapped phase
       phase_arr = np.stack([phase_freq1, phase_freq2, phase_freq3, phase_freq4])
       unwrap, k = nstep.multifreq_unwrap(pitch_list, phase_arr, kernel, direc)
       inte_img = cv2.imread(os.path.join(obj_path,'white.jpg'))
       if temp:
           temperature = np.load(os.path.join(obj_path,'temperature.npy'))
       else:
           temperature = 0
       inte_rgb = inte_img[...,::-1].copy()
       np.save(os.path.join(obj_path,'{}_obj_mod.npy'.format(type_unwrap)),mod_freq4) 
       np.savez(os.path.join(obj_path,'{}_unwrap.npz'.format(type_unwrap)),data = unwrap.data, mask = unwrap.mask) 
       obj_cordi, obj_color, obj_t, cordi_sigma = complete_recon(unwrap, inte_rgb, mod_freq4, limit, calib_path, sigma_path, phase_st, pitch_list[-1], N_list[-1], obj_path, temp, temperature)
       
   elif type_unwrap == 'multiwave':
       eq_wav12 = (pitch_list[-1] * pitch_list[1]) / (pitch_list[1]-pitch_list[-1])
       eq_wav123 = pitch_list[0] * eq_wav12 / (pitch_list[0] - eq_wav12)

       pitch_list = np.insert(pitch_list, 0, eq_wav123)
       pitch_list = np.insert(pitch_list, 2, eq_wav12)
       
       object_wav3, mod_wav3, avg_wav3, gamma_wav1, delta_deck_wav3 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(0, N_list[0])]), limit)
       object_wav2, mod_wav2, avg_wav2, gamma_wav2, delta_deck_wav2 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0], N_list[0] + N_list[1])]), limit)
       object_wav1, mod_wav1, avg_wav1, gamma_wav3, delta_deck_wav1 = nstep.mask_img(np.array([cv2.imread(os.path.join(obj_path,'capt_%d.jpg'%i),0) for i in range(N_list[0] + N_list[1], N_list[0]+ N_list[1]+ N_list[2])]), limit)

       #wrapped phase
       phase_wav1 = nstep.phase_cal(object_wav1, N_list[2], delta_deck_wav1 )
       phase_wav2 = nstep.phase_cal(object_wav2, N_list[1], delta_deck_wav2 )
       phase_wav3 = nstep.phase_cal(object_wav3, N_list[0], delta_deck_wav3 )
       phase_wav12 = np.mod(phase_wav1 - phase_wav2, 2 * np.pi)
       phase_wav123 = np.mod(phase_wav12 - phase_wav3, 2 * np.pi)       
       phase_wav123[phase_wav123 > TAU] = phase_wav123[phase_wav123 > TAU] - 2 * np.pi

       #unwrapped phase
       phase_arr = np.stack([phase_wav123, phase_wav3, phase_wav12, phase_wav2, phase_wav1])
       unwrap, k = nstep.multiwave_unwrap(pitch_list, phase_arr, kernel, direc)
       inte_img = cv2.imread(os.path.join(obj_path,'white.jpg'))    
       if temp:
           temperature = np.load(os.path.join(obj_path,'temperature.npy'))
       else:
           temperature = 0
       inte_rgb = inte_img[...,::-1].copy()
       np.save(os.path.join(obj_path,'{}_obj_mod.npy'.format(type_unwrap)),mod_freq4) 
       np.savez(os.path.join(obj_path,'{}_unwrap.npz'.format(type_unwrap)),data = unwrap.data, mask = unwrap.mask)
       obj_cordi, obj_color, obj_t, cordi_sigma = complete_recon(unwrap, inte_rgb, mod_wav3, limit,  calib_path, sigma_path, phase_st, pitch_list[-1], N_list[-1], obj_path, temp, temperature)   
   return obj_cordi, obj_color, obj_t, cordi_sigma
       

