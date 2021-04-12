#
# Copyright (c) Sinergise, 2019 -- 2021.
#
# This file belongs to subproject "field-delineation" of project NIVA (www.niva4cap.eu).
# All rights reserved.
#
# file in the root directory of this source tree.
# This source code is licensed under the MIT license found in the LICENSE
#

from typing import Callable, List, Any
from concurrent.futures import ProcessPoolExecutor

from fs_s3fs import S3FS
from dataclasses import dataclass
from tqdm.auto import tqdm

from sentinelhub import SHConfig


@dataclass
class BaseConfig:
    bucket_name: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str


def prepare_filesystem(config: BaseConfig) -> S3FS:
    return S3FS(bucket_name=config.bucket_name,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key,
                region=config.aws_region)


def set_sh_config(config: BaseConfig) -> SHConfig:
    """ Set AWS and SH credentials in SHConfig file to allow usage of download and io tasks """
    sh_config = SHConfig()

    sh_config.aws_access_key_id = config.aws_access_key_id
    sh_config.aws_secret_access_key = config.aws_secret_access_key

    if all(key in config.__annotations__.keys() for key in ['sh_client_id', 'sh_client_secret']):
        sh_config.sh_client_id = config.sh_client_id
        sh_config.sh_client_secret = config.sh_client_secret

    sh_config.save()

    return sh_config


def multiprocess(process_fun: Callable, arguments: List[Any], max_workers: int = 4) -> List[Any]:
    """
    Executes multiprocessing with tqdm.
    Parameters
    ----------
    process_fun: A function that processes a single item.
    arguments: Arguments with which te function is called.
    max_workers: Max workers for the process pool executor.

    Returns A list of results.
    -------

    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fun, arguments), total=len(arguments)))
    return results
