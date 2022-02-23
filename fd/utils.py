#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.
#

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from logging import Filter
from typing import Callable, List, Any, Iterable, Optional

from fs_s3fs import S3FS
from tqdm.auto import tqdm

from sentinelhub import SHConfig, CRS


@dataclass
class BaseConfig:
    bucket_name: Optional[str]
    aws_region: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]


def prepare_filesystem(config: BaseConfig, sh_config: Optional[SHConfig] = None) -> S3FS:
    aws_id = config.aws_access_key_id
    aws_secret = config.aws_secret_access_key

    if sh_config and all([sh_config.aws_access_key_id, sh_config.aws_secret_access_key]):
        aws_id = sh_config.aws_access_key_id
        aws_secret = sh_config.aws_secret_access_key

    return S3FS(bucket_name=config.bucket_name,
                aws_access_key_id=aws_id,
                aws_secret_access_key=aws_secret,
                region=config.aws_region, strict=False)


def set_sh_config(config: BaseConfig) -> SHConfig:
    """ Set AWS and SH credentials in SHConfig file to allow usage of download and io tasks """
    sh_config = SHConfig()

    if all(key in config.__annotations__.keys() for key in ['sh_client_id', 'sh_client_secret']):
        sh_config.sh_client_id = config.sh_client_id
        sh_config.sh_client_secret = config.sh_client_secret

    sh_config.save()

    return sh_config


def multiprocess(process_fun: Callable, arguments: Iterable[Any],
                 total: Optional[int] = None, max_workers: int = 4) -> List[Any]:
    """
    Executes multiprocessing with tqdm.
    Parameters
    ----------
    process_fun: A function that processes a single item.
    arguments: Arguments with which te function is called.
    total: Number of iterations to run (for cases of iterators) 
    max_workers: Max workers for the process pool executor.

    Returns A list of results.
    -------
    

    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fun, arguments), total=total))
    return results


class LogFileFilter(Filter):
    """ Filters log messages passed to log file
    """
    IGNORE_PACKAGES = (
        'eolearn.core',
        'botocore',
        'matplotlib',
        'fiona',
        'rasterio',
        'graphviz',
        'urllib3',
        'boto3',
        'tensorflow'
    )

    def filter(self, record):
        """ Shows everything from the main thread and process except logs from packages that are on the ignore list.
        Those packages send a lot of useless logs.
        """
        if record.name.startswith(self.IGNORE_PACKAGES):
            return False

        return record.threadName == 'MainThread' and record.processName == 'MainProcess'


def mgrs_to_utm(mgrs_tile_name: str) -> CRS:
    mgrs = mgrs_tile_name[:3]
    zone_id = int(mgrs[:2])
    north = mgrs[-1].upper() >= 'N'
    if north:
        base = 32600
    else:
        base = 32700
    return CRS(base + zone_id)
