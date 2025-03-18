import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Organoids Train Parser')
    parser.add_argument('-m', '--model', default='Unet', type=str, help="input model name",
                        choices=['Unet', 'FCN', 'Deeplab', 'Unet3+', 'Qnet','PSPnet','Uesnet50','Unet2+','ViT']
                        )
    parser.add_argument('--batch_size', '-b', default=64, type=int, help='Batch size for training')
    parser.add_argument('--epochs', '-e', default=10, type=int,help='total epochs for training')
    parser.add_argument('--input_height', default=224, type=int,help='input height for resize')
    parser.add_argument('--input_width', default=224, type=int,help='input width for resize')
    parser.add_argument('--data_path', default='./data/', type=str, help='dataset path')
    parser.add_argument('--init_lr', default=1e-3, type=float, help='initial lr')
    parser.add_argument('--max_lr', default=1e-3, type=float, help='max lr')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--n_classes', default=2, type=int, help='number of the classification types')
    # CommandLine.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int,help='number of workers for dataloader')
    parser.add_argument('-g', '--gpu', default=0, type=int, help="input the gpu num")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
