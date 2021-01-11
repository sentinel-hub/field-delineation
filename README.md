# Automatic field delineation

This repo contains code to generate automatic contours for agricultural parcels,
given Sentinel-2 images. The code uses data for Lithuania 2019, but same code 
can be used for any other country.

## Conda environment

Dependencies required are included in the [YAML](./delineation-gpu-env.yml) configuration file. To create a conda environment from this,
run

```bash
conda env create -f delineation-gpu-env.yml
```

The deep learning code uses `tensorflow` and our open-source collection [eo-flow](https://github.com/sentinel-hub/eo-flow), 
which needs to be installed.

## AWS set-up

To run the notebooks, an AWS S3 bucket need to be set-up as per [these instructions](https://docs.sentinel-hub.com/api/latest/#/BATCH_API/batch_processor?id=aws-s3-bucket-settings), 
to allow the batch API saving files to it. The credentials need to be either specified in notebooks or 
in environment variables.

```
bucket_name = "bucket-name"
aws_access_key_id = ""
aws_secret_access_key = ""
region = "eu-central-1"
```

The repo is currently divided into:

 * [input-data](#input-data)
 * notebooks
    * [data-download](#data-download)
    * [supervised](#supervised)
 

## Input data

In this folder we store the geometry for the area-of-interest in WGS84 coordinates. This geometry
will be split by batch API into smaller parallelizable patches. Here we also store
meta-data about the patches and training/validation/test patchlets. 
 
## Data download

The following [notebooks](./notebooks/data-download) deal with the following:

 * `01-data-download`: downloading the Sentinel-2 images (B-G-R-NIR) using Sentinel-Hub Batch API. For our use-case, we 
 downloaded all Sentinel-2 acquisitions within a time-period (3 months), with a given `maxcc` value. Since we wanted 
 data for 6 months, we repeated the process twice to cover the entire period. In your use-case, you might want to return
 a composite image (e.g. mosaick) over a given period of time instead of each single observation. This operation is specified in the `evalscript`. We can 
 help set this up if needed. Within this notebook, the batch API request is queried and monitored.
 * `02-tiffs-to-patches`: converting tiff files to eopatches. The tiff files saved in the bucket by the batch API are 
 read and converted to eopatches. In our case, two folders holding data for 3 months each are read and concatenated 
 temporally together. Adjust this notebook according to your use-case/`evalscript`. 
 * `03-vector-to-raster`: adding ground-truth vector data from a database to eopatches. Depending on the size and format
  of the reference GSAA vector data, a task is created to add this info to eopathces and to rasterize it for model training. 
  In our case, data is added to a database and added to the patches. Different processed masks are computed to train a 
  multi-task model. 
 * `04-add-cloud-masks`: process the downloaded `CLP` probabilities with `s2cloudless` post-processing to get the `CLM` 
 masks. This is not necessary and `CLM` masks can be downloaded directly from the service (as done for `CLP`).
 
After running these steps, eopatches with imaging and reference data should be available on the configured bucket.

## Supervised

The following [notebooks](./notebooks/supervised) deal with the following:

 * `01-patchlets-sampling`: sample patchlets of size `256x256` from the downloaded patches; 
 * `02-patchlets-to-npz-files`: create `.npz` files by aggregating patchlets in chunks (e.g. 2000). This facilitates 
 disk IO and RAM usage during training. Normalisation factors were computed from the eopatches and saved in a 
 separate `.csv` file, along with other meta-info (e.g. from which eopatch the patchlet was sampled from); 
 * `03-patchlets-split-train-cval-test`: split the patchlets into train/validation/test;
 * `04-train-model-from-cahced-npz`: train the [resunet-a model](https://arxiv.org/pdf/1910.12023.pdf). Modify the 
 config parameters for different behaviour, as well as the network architecture in `niva_models.py`;
 * `tf_data_utils` and `tf_viz_utils`: tensorflow helpers for dataset creation and visualization in `tensorboard`.
 
 
## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 776115.
