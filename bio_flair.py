from flair.data import Dictionary, Sentence
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.embeddings import FlairEmbeddings
import os
import shutil
import math
import codecs 
import flair
import torch

def run(pretrained_model, output_path):
    language_model = FlairEmbeddings(pretrained_model).lm

    # are you fine-tuning a forward or backward LM?
    is_forward_lm = language_model.is_forward_lm

    # get the dictionary from the existing language model
    dictionary: Dictionary = language_model.dictionary

    # get your corpus, process forward and at the character level
    corpus = TextCorpus('corpus/',
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # use the model trainer to fine-tune this model on your corpus
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(output_path,
                sequence_length=100,
                mini_batch_size=100,
                learning_rate=20,
                patience=10,
                checkpoint=True)

def create_partitions(filepath, n_train_partitions):
    if os.path.exists('corpus/'):
        shutil.rmtree('corpus/')
    os.makedirs('corpus/')
    os.makedirs('corpus/train')

    text = open(filepath, 'r', encoding = 'utf-8').read()
    lines = text.splitlines()
    
    n_examples = len(lines)
    n_train = math.floor(n_examples * 0.6)
    n_train_partition_size = math.floor(n_train / n_train_partitions)
    n_val = math.floor(n_examples * 0.2)
    n_test = math.floor(n_examples * 0.2)

    idx = 0
    for i in range(n_train_partitions):
        train = codecs.open(f'corpus/train/train_split_{i+1}.txt', 'w', 'UTF-8')
        for j in range(idx, idx+n_train_partition_size):
            train.write(lines[j]+'\n')
        idx+=n_train_partition_size
        train.close() 

    dev = codecs.open('corpus/valid.txt', 'w', 'UTF-8')
    for i in range(n_train, n_train+n_val):
        dev.write(lines[i] + '\n')
    dev.close()

    test = codecs.open('corpus/test.txt', 'w', 'UTF-8')
    for i in range(n_train+n_val, n_examples):
        test.write(lines[i] + '\n')  
    test.close() 


if __name__ == '__main__':
    n_train_partitions = 20
    available_gpu = torch.cuda.is_available()
    if available_gpu:
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        flair.device = torch.device(device)
    else:
        flair.device = torch.device('cpu')
    create_partitions('data/raw/ex.txt', 20)
    #create_partitions('data/raw/corpus_not_normalized.txt', 20)
    run('es-forward', 'resources/taggers/bio_flair_forward')
    run('es-backward', 'resources/taggers/bio_flair_backward')

    