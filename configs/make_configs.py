dataset = 'aligned_msa_filename: processed_cmx_uniref100_90_80_10_100.fasta\n'
threshold = 'reweighting_threshold: 0.8\n'
dataset_dirs = 'dataset_dir: sequence_sets\n'
working_dir = 'working_dir: output\n'
model_name = 'model_name: model\n'

hidden_layers = [200] #[1000, 500, 200]
latent_layers = [10, 5, 2] #[500, 100, 50, 40, 30, 20, 15, 10, 5, 2]
activation_func = 'activation_function: sigmoid\n'
learning_rate = 'learning_rate: .001\n'
epochs = 'epochs: 1000\n'
batch_size = 'batch_size: 128\n'
device = 'device: auto\n'

list_name = 'configs.txt'
config_list = open(list_name, 'w')

for hidden_layer in hidden_layers:
    for latent_layer in latent_layers:
        file_name = 'example_config_%s_%s.yaml' % (hidden_layer,latent_layer)
        hidden_text = 'hidden_layer_size: %s\n' % hidden_layer
        latent_text = 'latent_layer_size: %s\n' % latent_layer

        lines = [dataset, threshold, dataset_dirs, working_dir, model_name,\
                 hidden_text, latent_text, activation_func, learning_rate, epochs,\
                 batch_size, device]

        config = open(file_name, 'w')
        for line in lines: config.write(line)
        config_list.write('../configs/%s\n'%file_name)
        config.close()
config_list.close()
