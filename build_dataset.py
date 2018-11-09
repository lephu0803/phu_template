import argparse
import os
import glob
from PIL import Image
import random 
import tqdm

paser = argparse.ArgumentParser()
paser.add_argument('--input_dir', default='data/SIGN', help='Path to input data dir')
paser.add_argument('--output_dir', default='data/out_64x64_SIGN', help='Path to output data')
paser.add_argument('--output_size', default='64')

def resize_and_save(filename, output_dir, size):
    
    image = Image.open(filename)
    image = image.resize((size,size), Image.BILINEAR)
    image.save(os.path.join(output_dir,filename.split('/')[-1]))

if __name__ == '__main__':
    args = paser.parse_args()
    SIZE = args.output_size

    assert os.path.isdir(args.output_dir), "Couldnt find path {}".format(args.output_dir) 

    train_data_dir = os.path.join(args.data_dir, 'train')
    test_data_dir = os.path.join(args.data_dir, 'test')

    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    random.seed(200)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 *len(filenames))
    train_filenames = filenames[:split]
    dev_filenames =filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print('Warning: output dir {} already exists'.format(args.output_dir))

    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print('Warning: Output dir {} already exits'.format(output_dir_split))

        print('Processing {} data, saving preprocessing data to {}'.format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size = SIZE)
    print('Done building dataset')
