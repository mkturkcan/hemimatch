import json
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import cv2
import json
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

neuron_path = 'images/EM_Hemibrain11_0630_2020_radi2_PackBits_withNeuronName/'
onlyfiles = [f for f in listdir(neuron_path) if isfile(join(neuron_path, f))]

def match_gpu(file_path = 'my_example.png', path_dir = 'images/', batch_size = 32, workers=8):
    """Generate matches using GPU with Tensorflow.
    
    # Arguments:
        file_path (str): Optional. Path of the input file to use.
        path_dir (str): Optional. Path of the images folder to use.
        batch_size (int): Optional. Batch size to use. Must divide the input size.
        workers (int): Optional. Number of workers to use.
        
    # Returns:
        score_order: Order of ids, from maximum score to minimum.
        scores: The scores vector, aligned to the names.
        names: Names of the neurons.
    """
    import tensorflow as tf
    tf.keras.backend.clear_session()
    from tensorflow.keras import layers
    import tensorflow.keras.backend as K
    from keras.preprocessing.image import ImageDataGenerator
    import json
    import os
    from os import listdir
    from os.path import isfile, join
    from PIL import Image
    import numpy as np

    class Multiply(layers.Layer):
        def __init__(self, _in,  **kwargs):
            self._in = _in
            super(Multiply, self).__init__(**kwargs)
        def build(self, input_shape):
            self.WA = tf.convert_to_tensor( self._in, dtype = tf.float32 )
            self.built = True
        def call(self, inputs):
            in_a = inputs
            in_a = in_a[:,:,200:800,:]
            output = K.sum(K.batch_flatten(in_a * self.WA),-1)
            return output

    img_width, img_height = 566, 1210
    epochs = 50
    
    im = Image.open(file_path)
    imarray = np.array(im) / 255.
    imarray = np.repeat(np.reshape(imarray, (1,imarray.shape[0], imarray.shape[1], imarray.shape[2])), batch_size, axis=0)

    input_shape = (img_width, img_height, 3)

    x = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
    layer = Multiply(imarray)
    y = layer(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer, loss='mse')
    model.summary()
    model_datagen = ImageDataGenerator(rescale=1. / 255)

    model_generator = model_datagen.flow_from_directory(
        path_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    scores = model.predict_generator(model_generator, verbose=1, use_multiprocessing=True, workers=workers)
    score_order = np.argsort(scores)
    score_order = score_order[::-1]
    return score_order, scores, onlyfiles

def prep_binary_dataset():
    """Prepares a binary dataset to use for matches and saves it as match_X.npy.
    """
    images = []
    a = (len(onlyfiles), 566, 600)
    X = np.zeros((len(onlyfiles),a[1],a[2]), dtype=bool)
    for file_index in range(len(onlyfiles)):
        file_path = os.path.join(neuron_path,onlyfiles[file_index])
        # im = Image.open(file_path)
        im = cv2.imread(file_path)
        imarray = np.array(im)
        imarray = imarray[:,200:800,:]
        imarray = np.max(imarray,axis=2)
        X[file_index,:,:] = imarray>127
    np.save('match_X', X)
    
def load_binary_dataset():
    """Loads the binary dataset.
    
    # Arguments:
        np.array: The numpy array to use.
    """
    return np.load('match_X.npy')
    
def get_average_im(neuron_path = 'images/EM_Hemibrain11_0630_2020_radi2_PackBits_withNeuronName/'):
    """Generates the average image from all images in path.
    
    # Arguments:
        neuron_path (str): Path of the input images to use.
    """
    for file_index in range(len(onlyfiles)):
        file_path = os.path.join(neuron_path,onlyfiles[file_index])
        im = Image.open(file_path)
        imarray = np.array(im)
        if file_index == 0:
            average_im = 1. * imarray / 255.
        else:
            average_im = average_im + 1. * imarray / 255.
    average_im = average_im / len(onlyfiles)
    return average_im

def fix_contrast(x):
    """Returns the input array, limited to 0-1.
    
    # Arguments:
        x (np.array): A numpy array to use.
    """
    x = x - np.min(x)
    x = x / np.max(x)
    return x

def show_average_im(average_im):
    """Returns the input array, limited to 0-1.
    
    # Arguments:
        x (np.array): A numpy array to use.
    """
    import PIL.Image
    import numpy as np
    average_im_image = np.round(fix_contrast(average_im[:,200:800,:])*255.).astype(np.uint8)
    def imshow(img):
        import cv2
        import IPython
        _,ret = cv2.imencode('.jpg', img) 
        i = IPython.display.Image(data=ret)
        IPython.display.display(i)
    imshow(average_im_image)
    
    
def match_cpu_inmemory(file_path, X):
    """Generate matches using CPU.
    
    # Arguments:
        file_path (str): Path of the input file to use.
        X (np array): Precomputed image structure to use for matching.
        
    # Returns:
        score_order: Order of ids, from maximum score to minimum.
        scores: The scores vector, aligned to the names.
        names: Names of the neurons.
    """
    im = Image.open(file_path)
    imarray = np.array(im) * 1.
    input_im = np.max(imarray, axis=2) > 127
    I = np.repeat(np.reshape(input_im, (1,input_im.shape[0], input_im.shape[1])), X.shape[0], axis=0)
    scores = np.sum(np.sum(np.logical_and(I,X), axis=2), axis=1) - np.sum(np.sum(np.logical_and(~I,X), axis=2), axis=1)
    
    scores = np.array(scores)
    score_order = np.argsort(scores)
    score_order = score_order[::-1]
    return score_order, scores, onlyfiles

def match_cpu_sequential(file_path, neuron_path = 'images/EM_Hemibrain11_0630_2020_radi2_PackBits_withNeuronName/'):
    """Generate matches using CPU sequentially.
    
    # Arguments:
        file_path (str): Path of the input file to use.
        neuron_path (str): Path to the input images.
        
    # Returns:
        score_order: Order of ids, from maximum score to minimum.
        scores: The scores vector, aligned to the names.
        names: Names of the neurons.
    """
    onlyfiles = [f for f in listdir(neuron_path) if isfile(join(neuron_path, f))]
    im = Image.open(file_path)
    imarray = np.array(im) * 1.
    input_im = imarray / 255.

    names = []
    scores = []
    for file_index in range(len(onlyfiles)):
        names.append(onlyfiles[file_index])
        file_path = os.path.join(neuron_path,onlyfiles[file_index])
        im = Image.open(file_path)
        imarray = np.array(im) * 1.
        im_array = imarray[:,200:800,:] / 255.
        im_array = 1. * (im_array > 0)
        overlap = np.multiply(im_array, input_im)
        score = np.sum(overlap) - np.sum(np.multiply(im_array, 1.-input_im))
        scores.append(score)

    scores = np.array(scores)
    score_order = np.argsort(scores)
    score_order = score_order[::-1]
    return score_order, scores, names

def get_search(names, verb = 'show'):
    """Generates an NLP string given input.
    
    # Arguments:
        names (list): List of file names.
        verb (str): Optional. Verb to use for the query. Defaults to "show".
        
    # Returns:
        _str: String for the query.
    """
    _str = verb + ' /:referenceId:['+', '.join([str(i) for i in names])+']'
    print(_str)
    return _str

def get_top_k(names, score_order, K=50):
    """Get top K matches.
    
    # Arguments:
        names (list): List of names to use.
        score_order (array): Order of ids, from maximum score to minimum.
        K (int): Optional. K to use for retrieving matches. Defaults to 50.
        
    # str:
        scores: String for the query.
    """
    n_ids = [names[score_order[i]].split('-')[0] for i in range(K)]
    return get_search(n_ids)