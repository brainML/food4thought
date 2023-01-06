# food4thought

This repo includes the code and data required to replicate the results of the paper: "Selectivity for food in human ventral visual cortex". 

The NSD code is organized as such, with each file being independently runnable:
- Analysis
  - Encoding models
  - OLS encoding model
- Ridge regression encoding model
- Decoding models
  - Note: any of these steps can be replaced with custom searchlights
  - Searchlight_get_indices: get indices for each voxel’s searchlight
  - Searchlight_inds_to_voxel_vals: Using indices, get corresponding voxel values and filter voxel responses to be only that of relevant images (in our case, shared images)
  - Searchlight_cvm: run SVM cross validation on searchlight data to decode food v non food 
- PCA
- Clustering
  - Specify within this file what kind of input to cluster
- Data
  - Basic data needed that cannot be preprocessed by user
- Visualization
  - Example visualization tool 

Note: Voxel data is assumed to be inputted in a format of stimuli * voxel cortical surface

The localizer code is organized as such, with each file being independently runnable:
- analysis
  - run_all.ipynb used to run the localizer analysis for identifying food regions
  - A set of .py files containing the necessary analysis and utility functions
- data
  - Preprocessed localizer fMRI data for each subject along with the stimulus log files
- pycortex_db
  - Pycortex store for the food localizer subjects. Should be added to local pycortex store after pycortex installation
- food_images
  - The food images used in the food localizer
- res
  - Directory for storing the eventual results



