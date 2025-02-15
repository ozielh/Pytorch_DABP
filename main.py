import torch
import train
import mnist
import mnistm
import model
from utils import get_free_gpu, visualize_input

save_name = 'omg'
save_name_dann = 'dann'



def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()
        # 1 Original MNIST and MNIST-M inputs
        visualize_input()
        train.source_only(encoder, classifier, source_train_loader, target_train_loader, save_name)
        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name_dann)

    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    main()
