"""Script to run the dca.py program from outside the VAE repository

  Typical usage example: (from the command line)
	python dca_tools.py \
		-i ../sequence_sets/cmx_aligned_blank_90.fasta \
		-w ../working/cmx_aligned_blank_90_weights.npy
                -o ../working/cmx_dca_params.pkl

"""

import pickle
import dca

if __name__ == "__main__":
    import time
    import argparse
    import read_config
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--msa_filename",
                    help="input msa in ALN or FASTA format")
    parser.add_argument("-w", "--weights_filename",
                    help="Output filename for weights", default=None)
    parser.add_argument("-o", "--output_filename",
            help="Output filename for model params")
    parser.add_argument("-d", "--device",
                    help="Device to use", default="")
    parser.add_argument("-l", "--learning_rate",
            help="Learning rate for ADAM optimizer",
            type=float, default=0.01)
    parser.add_argument("-s", "--save_model_path",
            help="Path to save model state dictionary",
            default=None)
    parser.add_argument("-g", "--lossgraph_path",
            help="Path to save loss curve",
            default=None)
    args = parser.parse_args()

    if not args.device:
        device = read_config.get_best_device()
    else:
        device = args.device


    msa, msa_weights = load_full_msa_with_weights(
            msa_path=args.msa_filename,
            weights_path=args.weights_filename)
    ret = train_dca_model(device=device,
                       msa=msa, msa_weights=msa_weights,
                       learning_rate = args.learning_rate,
                       num_epochs=args.epochs)

    # save parameters
    
    with open(args.output_filename, 'wb') as fh:
        pickle.dump({k:ret[k] for k in ["weights", "bias"]}, fh) 

    if args.lossgraph_path
        # plot loss curve
        DCA.plot_loss_curve(losses=ret['losses'],  
                annotatation_str = str(ret['optimizer']),
                save_fig_path = args.lossgraph_path,
                model_name="")

    if args.save_model_path:
        # save model state
        torch.save(ret['model'].state_dict(), args.save_model_path)


     


