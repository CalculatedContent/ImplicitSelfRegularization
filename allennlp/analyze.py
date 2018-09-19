"""
The ``analyze`` subcommand can be used to
analyze a trained model against a dataset
and report any metrics calculated by the model.

.. code-block:: bash

    $ allennlp analyze --help
    usage: allennlp analyze [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file 

    Analyze the specified model + dataset

    positional arguments:
    archive_file          path to an archived trained model

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file to save metrics
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any, Iterable
import argparse
import logging
import json

import torch
import numpy as np
import powerlaw

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import prepare_environment
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Analyze(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Analyze the specified model + dataset'''
        subparser = parser.add_parser(
                name, description=description, help='Analyze the specified model + dataset')

        subparser.add_argument('archive_file', type=str, help='path to an archived trained model')

        subparser.add_argument('--output-file', type=str, help='path to output file')

        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.set_defaults(func=analyze_from_args)

        return subparser


def analyze(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    _warned_tqdm_ignores_underscores = False
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))
        for batch in generator_tqdm:
            batch = util.move_to_device(batch, cuda_device)
            model(**batch)
            metrics = model.get_metrics()
            if (not _warned_tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                logger.warning("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                _warned_tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        return model.get_metrics(reset=True)


def analyze_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    # named tuple, we only want pytorch model not config
    model = archive.model
    # pain .. can just save all matrices and load 

    for im, m in enumerate(model.modules()):
 #       print(im, m)
        if isinstance(m, torch.nn.Linear):
                
            W = np.array(m.weight.data.clone().cpu())
            M, N = np.min(W.shape), np.max(W.shape)
#            print("{}  SHAPE {} x {}".format(im,M,N))
            _, svals, _ = np.linalg.svd(W)
            minsval=np.min(svals)
            evals = svals*svals
            fit = powerlaw.Fit(evals)
            print("{} {} {} {} {} {}".format(im,M,N,fit.alpha, fit.D, minsval))
            
    return None
