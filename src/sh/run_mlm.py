"""Executing src/mlm/mlm.py for each number of layers.
"""
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--start_layer",
                    type=int,
                    default=1,
                    help="The starting number of layers")
parser.add_argument("--end_layer",
                    type=int,
                    default=48,
                    help="The end number of layers")
parser.add_argument("--gpus",
                    type=str,
                    default="0,1",
                    help="Which gpu to visible")
parser.add_argument("--init_readout",
                    action="store_true",
                    help="Whether or not to initialize readout layer.")
parser.add_argument(
    "--without_train",
    action="store_true",
    help="If without_train, just check the accuracy of ALBERT without training."
)
args = parser.parse_args()

DATA_DIR = "../data/wiki"
batch_size = 32
readout_type = "init_readout" if args.init_readout else "trained_readout"
for num_layer in range(args.start_layer, args.end_layer):
    for init_model in [True, False]:
        lr = 2e-3
        if init_model:
            flag = "init"
            epoch = 3
        else:
            flag = "pre-trained"
            epoch = 3
        cmd = [
            f"CUDA_VISIBLE_DEVICES={args.gpus}", "python",
            "src/script/calc_masked_language_modeling.py", "--model_type",
            "albert", "--model_name_or_path", "albert-large-v2", "--do_eval",
            "--do_lower_case", "--train_data_file",
            str(os.path.join(DATA_DIR,
                             "corpus_for_mlm_train.txt")), "--eval_data_file",
            str(os.path.join(DATA_DIR, "corpus_for_mlm_test.txt")), "--mlm",
            "--mlm_probability",
            str(0.15), "--block_size",
            str(128), "--per_gpu_eval_batch_size", "64",
            "--per_gpu_train_batch_size",
            str(batch_size), "--num_train_epochs",
            str(epoch), "--learning_rate",
            str(lr), "--overwrite_output_dir", "--num_layer",
            str(num_layer), "--fix_encoder", "--warmup_steps", "100",
            "--weight_decay", "0.001", "--save_steps", "1000000",
            "--logging_steps", "50", "--seed", "42"
        ]
        if init_model:
            cmd.append("--init_model")
        if args.init_readout:
            cmd.append("--init_readout")
        if not args.without_train:
            cmd.append("--do_train")
        cmd.append("--output_dir")
        if args.without_train:
            cmd.append(
                f"../out/mlm/fixed/{readout_type}/{flag}/{num_layer:02d}")
        else:
            cmd.append(
                f"../out/mlm/tuned/{readout_type}/{flag}/{num_layer:02d}")
        print(cmd)
        subprocess.call(" ".join(cmd), shell=True)
