# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Main tasks functionality."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron.initialize import initialize_megatron


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')
    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='Sliding window for overlapping evaluation.')
    group.add_argument('--strict-lambada', action='store_true',
                       help='Use more difficult formulation of lambada.')
    # Retriever args
    group.add_argument('--qa-data-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa-data-test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    # Faiss arguments for retriever
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--faiss-match', type=str, default='string', \
                        choices=['regex', 'string'], help="Answer matching '\
                        'logic type")
    group.add_argument('--faiss-topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    # finetune for retriever
    group.add_argument('--eval-micro-batch-size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch '
                            'size). Global batch size is local batch size '
                            'times data parallel size.')
    group.add_argument('--train-with-neg', action='store_true',
                       help='Whether to use negative examples during model '
                        'training')
    group.add_argument('--train-hard-neg', type=int, default=0,
                       help='Number of hard negative exmaples to use during '
                        'training')
    group.add_argument('--calibration-set', type=str, default='C4',
                          help='Calibration set for pruning')


    # parameters for Av.rank validation method
    # Following options/arguments have been taken directly from DPR codebase
    group.add_argument('--val-av-rank-hard-neg', type=int, default=30,
                        help='Av.rank validation: how many hard negatives to'
                        ' take from each question pool')
    group.add_argument('--val-av-rank-other-neg', type=int, default=30,
                        help='Av.rank validation: how many other negatives to'
                        ' take from each question pool')
    group.add_argument('--load-sparse-ckpt', default=None, type=str,
                        help='Path to the sparse model checkpoint')

    group.add_argument('--semi-structured', default=False, action='store_true',
                        help='Use semi-structured sparsity')
    group.add_argument('--target-layer', default=None, type=int,
                        help='Target layer for pruning, None for all layers')
    group.add_argument('--add-ste', default=None, action='store_true',
                        help='Add STE to the model')
    return parser


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_tasks_args)

    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    if args.task == 'RACE':
        from race.finetune import main
    elif args.task in ['MNLI', 'QQP']:
        from glue.finetune import main
    elif args.task in ['LAMBADA', 'WIKITEXT103', 'WIKITEXT2']:
        from zeroshot_gpt.evaluate import main
    elif args.task in ['ICT-ZEROSHOT-NQ', 'RETRIEVER-EVAL']:
        from orqa.evaluate_orqa import main
    elif args.task in ['RET-FINETUNE-NQ']:
        from orqa.supervised.finetune import main
    elif args.task in ['PRUNE-WIKITEXT103', 'PRUNE-WIKITEXT2']:
        from pruning.prune_main_llama import main
    elif args.task in ['PRUNE-SUBDOMAIN']:
        from pruning.prune_main_subdomain import main
    elif args.task in ['LATENCY']:
        from latency.test_latency import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main()
