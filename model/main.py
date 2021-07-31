from transformer_xl import Transformer_XL
import os
from glob import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    # declare model
    model = Transformer_XL(
        checkpoint='test-checkpoint-chord',
        is_training=True)
    # prepare data
    midi_paths = glob('../dataset/*.midi') # you need to revise it
    training_data = model.prepare_data(midi_paths=midi_paths)
    print(training_data)
    print("====================================================================")
    print("Training...")
    output_checkpoint_folder = 'test-output-chord' # your decision
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)
    
    # finetune
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)

    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into the your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model.close()

if __name__ == '__main__':
    main()