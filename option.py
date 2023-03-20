import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='SR', help='SR, RE')

# LF_SR
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--scale_factor", type=int, default=4, help="4, 2")

parser.add_argument('--model_name', type=str, default='RR_HLFSR', help="model name")
parser.add_argument("--use_pre_ckpt", type=bool, default=True, help="use pre model ckpt")
parser.add_argument("--path_pre_pth", type=str, default='./pretrained_model/RR_HLFSR_4xSR_5x5.pth.tar', help="path for pre model ckpt")
parser.add_argument('--data_name', type=str, default='ALL',
                    help='EPFL, HCI_new, HCI_old, INRIA_Lytro, Stanford_Gantry, ALL(of Five Datasets)')
parser.add_argument('--path_for_train', type=str, default='./data_for_training/')
parser.add_argument('--path_for_test', type=str, default='./data_for_test/')
parser.add_argument('--path_log', type=str, default='./log/')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--decay_rate', type=float, default=0, help='weight decay [default: 1e-4]')
parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument('--epoch', default=50, type=int, help='Epoch to run [default: 50]')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--parallel', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=2, help='num workers of the Data Loader')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=0 )
parser.add_argument('--save_path', type=str, default='../Results/')
parser.add_argument("--step_validation", type=int, default=15, help="number of epochs to perform validation")

# HLFSR network settings
parser.add_argument("--n_groups", type=int, default=10, help="number of HLFSR-Groups")
parser.add_argument("--n_blocks", type=int, default=15, help="number of SRBs blocks in one HFEM")
parser.add_argument("--channels", type=int, default=64, help="number of channels")
parser.add_argument("--crop_test_method",type=int, default=1, help="cropped test method( 1- whole image| 2- cropped mxn patches | 3- cropped 4 patches")

# disable calculate PSNR/SSIM 
parser.add_argument("--test_NTIRE2023_LFSR",type=int, default=0, help="disable calculate PSNR/SSIM since there is no ground truth image")

# enable self_ensemble 
parser.add_argument('--self_ensemble', action='store_true', default=True, help='use self-ensemble method for test')
parser.add_argument('--precision', type=str, default='single',choices=('single', 'half'), help='FP precision for test (single | half)')

args = parser.parse_args()



if args.task == 'SR':
    args.angRes_in = args.angRes
    args.angRes_out = args.angRes
    args.patch_size_for_test = 32
    args.stride_for_test = 16
    args.minibatch_for_test = 16

del args.angRes