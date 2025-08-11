#%% LOAD IMAGE
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import frangi
from skimage.exposure import equalize_adapthist
from skimage.segmentation import random_walker
import nibabel as nib
import numpy as np

patient= "sub-130"
path= "Données/"+patient+"/"+patient+"_stripped.nii"
denoising= "yes"

img_to_segment_nii = nib.load(path)
img_to_segment = img_to_segment_nii.get_fdata()

grey_img = img_to_segment/np.max(img_to_segment)

#%% DENOISING IMAGE

if denoising == "yes":
    sigma_est = np.mean(estimate_sigma(grey_img))
    img_denoised = denoise_nl_means(
                        grey_img, 
                        patch_size=30, 
                        patch_distance=60, 
                        h=2*sigma_est,
                        fast_mode=True)
elif denoising == "no":
    img_denoised = grey_img
    
#save image
img_denoised_nifti = nib.Nifti1Image(img_denoised, affine= img_to_segment_nii.affine)
nib.save(img_denoised_nifti, "Données/"+patient+"/"+patient+"_denoised")

CLAHE_img = equalize_adapthist(img_denoised, clip_limit=0.01)
#save image
CLAHE_img_nifti = nib.Nifti1Image(CLAHE_img, affine= img_to_segment_nii.affine)
nib.save(CLAHE_img_nifti, "Données/"+patient+"/"+patient+"_CLAHE")

#%% Vessel Enhancement frangi

path= "Données/"+patient+"/"+patient+"_denoised.nii"
img_denoised_nifti = nib.load(path)
img_denoised = img_denoised_nifti.get_fdata() 

img_VEnhanced = frangi(1-img_denoised, sigmas=[1,2,3,4,5])
img_VEnhanced_nifti = nib.Nifti1Image(img_VEnhanced, affine= img_denoised_nifti.affine)
nib.save(img_VEnhanced_nifti, "Données/"+patient+"/"+patient+"_VEnhanced_frangi")


#%% enhacement

img_CLAHE_nifti = nib.load("Données/"+patient+"/"+patient+"_CLAHE.nii")
img_frangi_nifti = nib.load("Données/"+patient+"/"+patient+"_VEnhanced_frangi.nii")
img_CLAHE = img_CLAHE_nifti.get_fdata()
img_frangi = img_frangi_nifti.get_fdata()

facteur=15
img_enhanced = np.clip(img_CLAHE + (img_frangi * facteur), 0, 1)

img_enhanced_nifti = nib.Nifti1Image(img_enhanced, affine= img_CLAHE_nifti.affine)
nib.save(img_enhanced_nifti, "Données/"+patient+"/"+patient+"_enhanced")
