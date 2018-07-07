
# coding: utf-8

# # Key notes
# 
# 
# 运行前请修改一下路径名或参数值：
# 
# [1]base_path = "./data/whale/train_full/" ，原始train image的存放路径
# 
# [2]data = pd.read_csv('./data/whale/train.csv')，train.csv的存放路径
# 
# [3]train_files = glob.glob("./data/whale/train_full/*.jpg")，原始train image的存放路径
# 
# [4]test_files = glob.glob("./data/whale/test/*.jpg")，原始test image的存放路径
# 
# [5]最后会在当前目录下生成一个sub_triplet_loss.csv预测文件，即kaggle规定的submission文件
# 
# [6]如果laptop运行，GPU内存不到8G，有可能会发生OOM溢出，可以适当减少batch_size
# 
# 
# 
# 1.主框架模型时siamese + triplet loss, 求每个image的embedding（长度为50的vector）时用了resNet50模型
# 
# 2.triplet loss使用了Bayesian Personalized Ranking loss
# 
# 3.预处理只做了resize to (256, 256)和convert to RGB, 以及小概率(10%)的fliplr，没有做augment
# 
# 4.挑选triplet的时候没有刻意去挑选（像andrew ng的video和那篇paper里面说的那样，需要刻意去挑选，以提升性能。
# 
# 5.在生产训练batch的时候，用了generator + yield，所有batch的生成都是on the fly，大大减少了memory的消耗
# 
# 6.模型训练完之后，将所有的train images和test images的embedding都计算出来，在embedding（长度只有50）的基础上再做test images的预测，这样大大提升了预测速度
# 
# 7.预测用了knn算法，最后选取最近的5个class id
# 
# 8.因为new whale有大概率出现（9850个train images出现了810次），这里强制把new whale在knn的distance设置为0.1（有待商榷）
# 
# 总结：最粗糙暴力的方法，最后的MAP有0.42左右，排名还挺高，可以考虑在这个基础上优化。
# 
# 优化方向：
# 
# 1.preprocessing
# 
# 2.augmentation
# 
# 3.the way to choose triplet
# 
# 4.the knn default distance for new_whale 0.1 ??
# 
# 5.make use of the test images, for example autoencoder


#part of the code is from https://github.com/maciejkula/triplet_recommendations_keras

from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Flatten, Input, merge
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, GlobalMaxPooling2D
import glob
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization,     GlobalMaxPool2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.applications.resnet50 import ResNet50
from sklearn.neighbors import NearestNeighbors  
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[20]:


class sample_gen(object):
    def __init__(self, file_class_mapping, other_class = "new_whale"):
        self.file_class_mapping= file_class_mapping
        self.class_to_list_files = defaultdict(list)
        self.list_other_class = []
        self.list_all_files = list(file_class_mapping.keys())
        self.range_all_files = list(range(len(self.list_all_files)))

        for file, class_ in file_class_mapping.items():
            if class_ == other_class:
                self.list_other_class.append(file)
            else:
                self.class_to_list_files[class_].append(file)

#       注意这里的class id有重复
        self.list_classes = list(set(self.file_class_mapping.values()))
        self.range_list_classes= range(len(self.list_classes))
#       每个class（Id）的比重，相当于直方图  
        self.class_weight = np.array([len(self.class_to_list_files[class_]) for class_ in self.list_classes]) * 1.0
#         self.class_weight = self.class_weight/np.sum(self.class_weight)
        
        self.class_weight /= self.class_weight.sum()
        print ("sum=", self.class_weight.sum())

#   这个函数只是返回一个triplet样例
    def get_sample(self):
#       按class id比重抽取一个样本
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]
#       对这种class id的，抽取两个样本images (如果某个class只有一个样本，那么返回的是两个一样的image)
        examples_class_idx = np.random.choice(range(len(self.class_to_list_files[self.list_classes[class_idx]])), 2)
#       注意这两个样本属于同一个class
        positive_example_1, positive_example_2 =             self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[0]],            self.class_to_list_files[self.list_classes[class_idx]][examples_class_idx[1]]

#       提取一个跟positive_example_1不同class的样本
        negative_example = None
        while negative_example is None or self.file_class_mapping[negative_example] ==                 self.file_class_mapping[positive_example_1]:
            negative_example_idx = np.random.choice(self.range_all_files, 1)[0]
            negative_example = self.list_all_files[negative_example_idx]
        return positive_example_1, negative_example, positive_example_2



# 就是返回了y_pred的平均值
def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)

# Bayesian Personalized Ranking loss
def bpr_triplet_loss(X):

    positive_item_latent, negative_item_latent, user_latent = X

    # BPR loss
    loss = 1.0 - K.sigmoid(
        K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
        K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))

    return loss

def euclid_triplet_loss(X, margin=0.3):
    
    positive_item_latent, negative_item_latent, user_latent = X
    
    #Euclid distance triplet loss
    loss = K.maximum(K.sum(K.square(user_latent - positive_item_latent), axis=-1, keepdims=True) 
                     - K.sum(K.square(user_latent - negative_item_latent), axis=-1, keepdims=True) + margin, 0)
    
    return loss

def get_base_model():
    latent_dim = 128
#   include_top：whether to include the fully-connected layer at the top of the network.
    base_model = ResNet50(weights='imagenet',include_top=False) # use weights='imagenet' locally

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dropout(0.5)(x)
    dense_1 = Dense(latent_dim)(x)
    normalized = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dense_1)
#   相当于对这50长度的vector，每个元素取平方，方便后面的距离计算
    base_model = Model(base_model.input, normalized, name="base_model")
    return base_model

def build_model():
    base_model = get_base_model()
#   input结构变成(256, 256, 3)
    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

#   用triplet loss的方式对三个embedding进行merge,输出是一个sigmoid
    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=euclid_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))

    print (model.summary())

    return model


def build_inference_model(weight_path):#=file_path):
    base_model = get_base_model()

    positive_example_1 = Input(input_shape+(3,) , name='positive_example_1')
    negative_example = Input(input_shape+(3,), name='negative_example')
    positive_example_2 = Input(input_shape+(3,), name='positive_example_2')

    positive_example_1_out = base_model(positive_example_1)
    negative_example_out = base_model(negative_example)
    positive_example_2_out = base_model(positive_example_2)

    loss = merge(
        [positive_example_1_out, negative_example_out, positive_example_2_out],
        mode=bpr_triplet_loss,
        name='loss',
        output_shape=(1, ))

    model = Model(
        input=[positive_example_1, negative_example, positive_example_2],
        output=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.000001))

#   导入前面训练出来的权重
    model.load_weights(weight_path)

#   base model只包含了把input转为embedding的过程，没有包含后面的triplet loss部分
    inference_model = Model(base_model.get_input_at(0), output=base_model.get_output_at(0))
    inference_model.compile(loss="mse", optimizer=Adam(0.000001))
    print (inference_model.summary())

    return inference_model

def read_and_resize(filepath):
#   这里不是用的grayscale，而是转成RGB了
    im = Image.open((filepath)).convert('RGB')
    im = im.resize(input_shape)
#   im的shape变成（256， 256， 3）
    im_array = np.array(im, dtype="uint8")[..., ::-1] #这个是对RGB进行逆序？？
#   转换成float类型
    return np.array(im_array / (np.max(im_array)+ 0.001), dtype="float32")

# 进行小概率的augment
def augment(im_array):
    if np.random.uniform(0, 1) > 0.9:
#       fliplr只对第1维度column进行flip
        im_array = np.fliplr(im_array)
    return im_array


# 进行大概率的augment，更复杂
def augment_v2(im_array):
    if np.random.uniform(0, 1) > 0.5:
        im_array = datagen.random_transform(im_array)
    return im_array

# 这个函数返回一个generator
def gen(triplet_gen):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []

#       会有重复抽样 bootstrap
        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            positive_example_1_img, negative_example_img, positive_example_2_img = read_and_resize(base_path+positive_example_1),                                                                        read_and_resize(base_path+negative_example),                                                                        read_and_resize(base_path+positive_example_2)
#           这个增强并没有增加训练样本数，而是替换了原样本
            positive_example_1_img, negative_example_img, positive_example_2_img = augment_v2(positive_example_1_img),                                                                                    augment_v2(negative_example_img),                                                                                    augment_v2(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        list_positive_examples_1 = np.array(list_positive_examples_1)
        list_negative_examples = np.array(list_negative_examples)
        list_positive_examples_2 = np.array(list_positive_examples_2)
        
#       利用yield，返回一个generator, 并且call on the fly (通过yield + while True)，节省内存
#       注意配合model.fit_generator使用的generator返回值必须是（input, target），所以后面的np.ones(batch_size)相当于target (即label)
#       只不过在这个模型里面这个target没有被用上而已
#       最后注意每次yield返回一个batch的samples
        yield [list_positive_examples_1, list_negative_examples, list_positive_examples_2], np.ones(batch_size)

    
# 这个函数返回一个generator
# 使用了“FaceNet: A Unified Embedding for Face Recognition and Clustering” 推荐的online hard triplets selection方法
# 用一个latest best model来做筛选，但只对hard negative进行筛选 (semi-hard)
def gen_with_online_selection(triplet_gen):
    while True:
        list_positive_examples_1 = []
        list_negative_examples = []
        list_positive_examples_2 = []
        
        
        latest_best_model = build_inference_model()
#       会有重复抽样
        for i in range(batch_size):
            positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
            positive_example_1_img, negative_example_img, positive_example_2_img = read_and_resize(base_path+positive_example_1),                                                                        read_and_resize(base_path+negative_example),                                                                        read_and_resize(base_path+positive_example_2)
#           这个增强并没有增加训练样本数，而是替换了原样本
            positive_example_1_img, negative_example_img, positive_example_2_img = augment_v2(positive_example_1_img),                                                                                    augment_v2(negative_example_img),                                                                                    augment_v2(positive_example_2_img)

            list_positive_examples_1.append(positive_example_1_img)
            list_negative_examples.append(negative_example_img)
            list_positive_examples_2.append(positive_example_2_img)

        list_positive_examples_1 = np.array(list_positive_examples_1)
        list_negative_examples = np.array(list_negative_examples)
        list_positive_examples_2 = np.array(list_positive_examples_2)
        
#       利用yield，返回一个generator, 并且call on the fly (通过yield + while True)，节省内存
#       注意配合model.fit_generator使用的generator返回值必须是（input, target），所以后面的np.ones(batch_size)相当于target (即label)
#       只不过在这个模型里面这个target没有被用上而已
#       最后注意每次yield返回一个batch的samples
        yield [list_positive_examples_1, list_negative_examples, list_positive_examples_2], np.ones(batch_size)



def data_generator(fpaths, batch=16):
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        img = read_and_resize(path)
        imgs.append(img)
#       获取image的名字
        fnames.append(os.path.basename(path))
        if i == batch:
            i = 0
            imgs = np.array(imgs)
#           每次yield返回一个batch的samples
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        yield fnames, imgs
    raise StopIteration()


"""
Driver
"""
if __name__ == "__main__":
    batch_size = 4
    input_shape = (256, 256)
    base_path = "../data/train/"
    
    model_name = "triplet_model"
    file_path = model_name + "weights.best.hdf5"

    datagen_args = dict(rotation_range=10,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

    datagen = ImageDataGenerator(**datagen_args)
    
    num_epochs = 300

    # Read data
    data = pd.read_csv('../data/train.csv')
    train, test = train_test_split(data, test_size=0.3, shuffle=True, random_state=1337)
    #把image作为key，id作为value
    file_id_mapping_train = {k: v for k, v in zip(train.Image.values, train.Id.values)}
    file_id_mapping_test = {k: v for k, v in zip(test.Image.values, test.Id.values)}
    train_gen = sample_gen(file_id_mapping_train)
    test_gen = sample_gen(file_id_mapping_test)
    
    
    # Prepare the test triplets
    model = build_model()
    
    #model.load_weights(file_path)
    
    # 根据monitor的值即loss，保存loss最小(min)时的model (best model)
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=4, save_best_only=True, mode='min')
    
    early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    
    callbacks_list = [checkpoint, early]  # early
    
    # Trains the model on data generated batch-by-batch by a Python generator
    # 这种模式，generate bath on the fly，可以节省很多memory，因而可以使用更大的batch size
    history = model.fit_generator(gen(train_gen), validation_data=gen(test_gen), epochs=num_epochs, verbose=4, workers=1, #use_multiprocessing=False,
                                  callbacks=callbacks_list, steps_per_epoch=500, validation_steps=50)
                                  
    
    model_name = "triplet_loss"
    
    data = pd.read_csv('./data/whale/train.csv')
    file_id_mapping = {k: v for k, v in zip(data.Image.values, data.Id.values)}
    
    inference_model = build_inference_model()
    
    # 文件名匹配，返回一个list包含所有这个后缀的文件path
    train_files = glob.glob("./train/*.jpg")
    test_files = glob.glob("./test/*.jpg")
    
    train_preds = []
    train_file_names = []
    i = 1
    # 每个imgs里面包含的是一个batch的samples
    for fnames, imgs in data_generator(train_files, batch=32):
    #     print (i*32/len(train_files)*100)
        i += 1
        predicts = inference_model.predict(imgs)
    #   将一个batch的images转换成embeddings，然后转成list
        predicts = predicts.tolist()
        train_preds += predicts
        train_file_names += fnames
    
    #  得到了所有train images的embeddings
    train_preds = np.array(train_preds)
    
    test_preds = []
    test_file_names = []
    i = 1
    for fnames, imgs in data_generator(test_files, batch=32):
    #     print (i * 32 / len(test_files) * 100)
        i += 1
        predicts = inference_model.predict(imgs)
        predicts = predicts.tolist()
        test_preds += predicts
        test_file_names += fnames
    
    #  得到了所有test images的embeddings
    test_preds = np.array(test_preds)
    
    # 这里用欧式距离判断class id，并且选取了6个neighbors
    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(train_preds)
    #distances, neighbors = neigh.kneighbors(train_preds)
    
    #print (distances, neighbors)
    
    # 对每个test样本，返回最近的六个embeddings,注意neighbors_test是train_preds里面样本的Index，而非样本本身
    distances_test, neighbors_test = neigh.kneighbors(test_preds)
    
    distances_test, neighbors_test = distances_test.tolist(), neighbors_test.tolist()
    
    preds_str = []
    
    for filepath, distance, neighbour_ in zip(test_file_names, distances_test, neighbors_test):
        sample_result = []
        sample_classes = []
        for d, n in zip(distance, neighbour_):
            train_file = train_files[n].split(os.sep)[-1]
            class_train = file_id_mapping[train_file]
            sample_classes.append(class_train)
            sample_result.append((class_train, d))
    
        if "new_whale" not in sample_classes:
            sample_result.append(("new_whale", 0.05))#new_whale有大概率出现，距离设置为0.1
        sample_result.sort(key=lambda x: x[1])
        sample_result = sample_result[:5] #取前五个距离最小的预测值
        preds_str.append(" ".join([x[0] for x in sample_result]))
    
    df = pd.DataFrame(preds_str, columns=["Id"])
    df['Image'] = [x.split(os.sep)[-1] for x in test_file_names]
    df.to_csv("sub_%s.csv"%model_name, index=False)
    
    # check the distance range
    # 如何选择new_whale对应的距离值？  why 0.1 ？？？？
    test_preds_parts = test_preds[:10]
    distances_test, neighbors_test = neigh.kneighbors(test_preds_parts)
    print (distances_test)
    
    