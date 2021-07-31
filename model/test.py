from transformer_xl import Transformer_XL
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    # declare model
    model = Transformer_XL(
        checkpoint='test-checkpoint-chord',
        is_training=False)
    
    # generate from scratch
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path='./result/from_scratch.midi',
        emotion='negative',
        prompt=None)
    
    # close model
    model.close()

if __name__ == '__main__':
    main()