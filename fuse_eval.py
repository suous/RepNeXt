import argparse
import torch

from data.datasets import build_dataset
from engine import evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('RepNeXt fused model evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=256, type=int)

    # Model parameters
    parser.add_argument('--model', default='repnext_m1', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='/root/FastBaseline/data/imagenet', type=str, help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'], type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name', choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'], type=str, help='semantic granularity')
    
    # Work parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    assert len(args.resume) > 0, "must provide a resume path or url."
    device = torch.device(args.device)

    dataset_val, _ = build_dataset(is_train=False, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print("Loading local checkpoint at {}".format(args.resume))
    model = torch.jit.load(args.resume, map_location='cpu')

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    print(f"Evaluating model: {args.model}")
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RepNeXt fused model evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
