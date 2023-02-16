# food4thought

This repo includes the visual food localizer, the code and the links to data required to replicate the results of the paper: ["Selectivity for food in human ventral visual cortex"](https://www.nature.com/articles/s42003-023-04546-2). 

## Localizer

The visual food localizer is built on the fLoc localizer published in [Stigliani et al., 2015](https://www.jneurosci.org/content/35/36/12412). To run this localizer:
1. Download and install the [fLoc localizer](https://github.com/VPNL/fLoc) following the instructions provided. Make sure the localizer is running as expected.
2. Copy the food image directory `food4thought/localizer/food` from this repository into the `fLoc/stimuli` directory in the fLoc repository. I.e., there needs to be a new directory named `food` along with the other directories (`adult`,`body`,`car`, etc.).
3. Add the line `stim_set3 = {'food','body' 'word' 'adult', 'house'};` after the line 24 of `fLoc/functions/fLocSequence.m`. Change the new line 27 to `stim_per_set = 80;` (this will change the localizer to only use the first 80 images of each category). Add the following case to the `run_sets` function after line 107:
```
    case 4
        run_sets = repmat(seq.stim_set3, seq.num_runs, 1);
```
4. Change the while loop on line 42 of `fLoc/runme.m` to:
```
    while ~ismember(stim_set, 1:4)
        stim_set = input('Which stimulus set? (1 = standard, 2 = alternate, 3 = both, 4 = food) : ');
    end
```
5. When running the localizer, specify option 4. The run should take 4 min exactly.

## Localizer data analysis
### Method 1
Follow the [fLoc instructions](https://github.com/VPNL/fLoc#analysis) to analyse the data using vistasoft. 

### Method 2
The preprocessed localizer data from our paper is available at [this link](https://kilthub.cmu.edu/articles/dataset/Selectivity_for_food_in_human_ventral_visual_cortex/22049177). After downloading it can be added to the `food4thought/localizer/data` directory. Please contact the corresponding author to obtain the pycortex store for the subjects from the paper. After you obtain them, you should add them to your [pycortex store](https://gallantlab.github.io/pycortex/). You can also run this analysis with your own data, after preprocessing it, obtaining the Free Surfer surfaces for your subject and adding them to pycortex.

The localizer code is organized as such, with each file being independently runnable:
- analysis
  - run_all.ipynb used to run the localizer analysis for identifying food regions
  - A set of .py files containing the necessary analysis and utility functions
- data
  - Preprocessed localizer fMRI data for each subject along with the stimulus log files
- food
  - The food images used in the food localizer
- res
  - Directory for storing the eventual results

## NSD data analysis
The NSD data was collected by [Allen et al., 2021](https://www.nature.com/articles/s41593-021-00962-x) and is freely available at http://naturalscenesdataset.org/. The NSD code is organized as such, with each file being independently runnable:
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



