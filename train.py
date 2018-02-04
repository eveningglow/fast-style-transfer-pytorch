from solver import Solver
import argparse

def main(args):
    s = Solver(trn_dir = args.trn_dir, 
               style_path = args.style_path, 
               result_dir = args.result_dir, 
               weight_dir = args.weight_dir,
               num_epoch = args.num_epoch,
               batch_size = args.batch_size,
               content_loss_pos = args.content_loss_pos,
               lr = args.lr,
               lambda_c = args.lambda_c,
               lambda_s = args.lambda_s,
               show_every = args.show_every,
               save_every = args.save_every)
    
    s.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trn_dir', type=str, default='data/train',
                        help='Training data directory')
    parser.add_argument('--style_path', type=str, default='style/abstract_1.png',
                        help='Style image path')
    parser.add_argument('--result_dir', type=str, default='check',
                        help='Result image directory for checking')
    parser.add_argument('--weight_dir', type=str, default='weight',
                        help='Weight of model directory')
    parser.add_argument('--num_epoch', type=int, default=2,
                        help='Training epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--content_loss_pos', type=int, default=1,
                        help='0 : relu_1_2 / 1 : relu_2_2 / 2 : relu_3_3 / 3 : relu_4_3')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--lambda_c', type=float, default=1,
                        help='Weight for content loss')
    parser.add_argument('--lambda_s', type=float, default=5e+5,
                        help='Weight for style loss')
    parser.add_argument('--show_every', type=int, default=500,
                        help='How often do you want to check result?')
    args = parser.parse_args()
    main(args)
    