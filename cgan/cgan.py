from __future__ import print_function, division




from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np
import sys
import os

# https://github.com/miranthajayatilake/CGAN-Keras
# https://medium.com/@utk.is.here/training-a-conditional-dc-gan-on-cifar-10-fce88395d610
# https://github.com/r0nn13/conditional-dcgan-keras
# https://arxiv.org/pdf/1610.09585.pdf
# https://arxiv.org/pdf/1411.1784.pdf
# http://cs231n.stanford.edu/reports/2017/pdfs/316.pdf
# https://eccv2018.org/openaccess/content_ECCV_2018/papers/Xinyuan_Chen_Attention-GAN_for_Object_ECCV_2018_paper.pdf
# https://openreview.net/forum?id=rJedV3R5tm
# http://bmvc2018.org/contents/papers/0247.pdf
# https://www.researchgate.net/publication/324783775_Text_to_Image_Synthesis_Using_Generative_Adversarial_Networks
# https://skymind.ai/wiki/generative-adversarial-network-gan
# https://antonia.space/text-to-video-generation
# https://hci.iwr.uni-heidelberg.de/system/files/private/downloads/1009852523/frank_gabel_eml2018_report.pdf
# https://www.groundai.com/project/dm-gan-dynamic-memory-generative-adversarial-networks-for-text-to-image-synthesis/
# https://www.topbots.com/ai-research-generative-adversarial-network-images/
# https://papers.nips.cc/paper/7290-text-adaptive-generative-adversarial-networks-manipulating-images-with-natural-language.pdf
# http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf
# http://proceedings.mlr.press/v48/reed16.pdf
# https://codeburst.io/understanding-attngan-text-to-image-convertor-a79f415a4e89
# https://github.com/zsdonghao/text-to-image
# https://github.com/crisbodnar/text-to-image


# https://github.com/kcct-fujimotolab/StackGAN
# https://arxiv.org/pdf/1612.03242.pdf


# http://cican17.com/gan-from-zero-to-hero-part-2-conditional-generation-by-gan/
# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
# https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master/chapter4-gan

scriptpath = "cgan/shapes_generator.py"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(scriptpath))
from shapes_generator import generateLabeledDataset

class CGAN():
    def __init__(self, imageShape, labelsShape):
        # Input shape
        height, width, depth = imageShape
        
        self.img_rows = height
        self.img_cols = width
        self.channels = depth
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # self.num_classes = 10
        self.num_shapes = labelsShape[0]
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        # label = Input(shape=(1,), dtype='int32')
        label = Input(shape=(self.num_shapes,), dtype='float32')

        label_embedding = Flatten()(Embedding(self.num_shapes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        # label = Input(shape=(1,), dtype='int32')
        label = Input(shape=(self.num_shapes,), dtype='float32')

        label_embedding = Flatten()(Embedding(self.num_shapes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)


    def trainOnDataset(self, X_train, y_train, epochs, batch_size=128, sample_interval=50):

        # # Load the dataset
        # (X_temp, y_temp), (_, _) = mnist.load_data()
        # X_temp = np.expand_dims(X_temp, axis=3)
        # y_temp = y_temp.reshape(-1, 1)
        # X_train = datasetX
        # y_train = datasetY

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)
        # y_train = y_train.reshape(-1)
        y_train = y_train / np.max(y_train)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    # cgan.train(epochs=20000, batch_size=32, sample_interval=200)
    images, labels = generateLabeledDataset(100, (32,32), (3,3))

    cgan = CGAN(images[0].shape, labels[0].shape)


    cgan.trainOnDataset(images, labels, 100)

    
