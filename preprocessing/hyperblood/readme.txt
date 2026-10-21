ITAI PAS HyperBlood: HSI blood detection & classification dataset
authors: M. Romaszewski, P.Głomb, M. Cholewa, A. Sochan 
Institute of Theoretical and Applied Informatics, Polish Academy of Sciences  (ITAI PAS) https://www.iitis.pl

The dataset is licensed under the Creative Commons Attribution 4.0 International License
To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/

Contact:
Michał Romaszewski, mromaszewski@iitis.pl, http://iitis.pl/en/person/mromaszewski
Przemysław Głomb, pglomb@iitis.pl, https://www.iitis.pl/pl/person/pglomb

Short description:

This dataset contains 14 hyperpsectral images of a mock-up scene with annotated pixels where blood and blood-like substances are visible. The intention behind the dataset is to facilitate the creation of ML detection and classification algorithms. The majority of images were captured with VNIR SOC-710 camera with spectral range 377-1046nm and 128 spectral bands. 

Directory structure:
	* data - hyperspectral images in ENVI format (.float and .hdr pairs)
	* anno - annotation files (numpy arrays)
	* images - simple class visualisation
	* src - API for loading images (python 3.6.9)

Notes:
	* Values in datacubes are a corrected reflectance - the correction algorithm details are in the corresponding paper. 
	* HSI images are annotated with letters (see paper)
		+In the /images directory, an image 'mock_up_scene.png' presents the location of each letter except "F" - the Frame image, which is presented in "frame_image.png"
		+File names are in format: <code letter>_<day of acquisition> e.g. F_1 for the "Frame" image from day 1. In the paper a code F(1) is used.
	*A loading API is provided in the src/ds_load.py (minmal loading code below) along with a simple demo: hyperpsectral classification of an example image with the SVM classifier
		+A subset of bands is noisy (see paper), they are removed by API functions by default (leaving 113 bands).
	
Classes:
    0 - background
    1 - blood
    2 - ketchup
    3 - artificial blood
    4 - beetroot juice
    5 - poster paint
    6 - tomato concentrate
    7 - acrtylic paint
    8 - uncertain blood
	
The dataset is described in detail in the papper "A Dataset for Evaluating Blood Detection in Hyperspectral Images". 
The arxive preprint is available online:
	
@misc{romaszewski2020dataset,
	title={A Dataset for Evaluating Blood Detection in Hyperspectral Images},
	author={Micha{\l} Romaszewski and Przemys{\l}aw Głomb and Arkadiusz Sochan and Micha{\l} Cholewa},
	year={2020},
	eprint={2008.10254},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	note={arXiv preprint arXiv:2008.10254},
}


Minimal leading code (using /src/ds_load.py):

from ds_load import get_data,get_anno

name ='F(1)'
data,wav = get_data(name)
anno = get_anno(name)
