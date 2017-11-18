
# coding: utf-8

# # 并联网络
# 
# 仿照ResNet, 建立并联网络. 
# 
# 其中并联网络可以是: 
# * 主路径为全连接网络, 捷径为identity或全连接网络
# * 主路径为Conv2d网络, 捷径为identiy或Conv2d网络
# * \>=2条路径
# * 必要时并联网络可以级联
# 
# 本程序中大量参考deeplearning.ai第4门课中第2周的作业ResNet

# In[1]:


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization
from keras.layers import Flatten, Reshape
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
get_ipython().magic('matplotlib inline')

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# # 并联
# 仿照并联电路的设定, 数据X同时流向多条路径, 经过每条路径的处理后再汇集到一起

# In[ ]:





# # 分支路径举例

# ## 短路路径

# In[2]:


def identify_path(X):
    return X


# ## 全连接路径

# In[3]:


def FC_path(X,dense_list,act_list):
    # 获取原shape
    origin_shape=X.get_shape().as_list()[1:]

    # 展平主路径, 如果不是m,n_H,n_W,n_C形式的, 则不需要使用Flatten
    if len(origin_shape)>=3:
        X = Flatten()(X) # Flatten

    # 主路径添加全连接层
    for stage in range(len(dense_list)):
        d=dense_list[stage]
        if d == None:
            d=np.prod(origin_shape)
        act=act_list[stage]
        X = Dense(d, 
              activation=act, name='fc_main_' + str(stage), 
              kernel_initializer = glorot_uniform(seed=0))(X)
    X = Reshape(origin_shape)(X)
    return X  


# ## 卷积路径

# In[4]:


def conv2d_block(X,f, filter_list, kernel_list,stride_list,padding_list,act_list):
    
    for idx in range(len(filter_list)):
    # First component of main path
        X = Conv2D(filters = filter_list[idx], 
                   kernel_size =kernel_list[idx], 
                   strides = stride_list[idx], 
                   padding = padding_list[idx], 
                   kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = -1)(X)
        X = Activation(act_list[idx])(X)
    return X

def conv2d_3_path(X,f,s,filter_list):
    kernel_list=[(1,1),(f,f),(1,1)]
    stride_list=[(s,s),(1,1),(1,1)]
    padding_list=['valid','same','valid']
    act_list=['relu','relu',None]
    return conv2d_block(X,f, filter_list, kernel_list,stride_list,padding_list,act_list)

def conv2d_1_path(X,f,s,filter_list):
    filters =[(filter_list[-1])]
    kernel_list=[(1,1)]
    stride_list=[(s,s)]
    padding_list=['valid']
    act_list=[None]
    return conv2d_block(X,f, filters, kernel_list,stride_list,padding_list,act_list)


# # 并联
# * identify_FC_block: 并联短路与全连接路径
# * identify_conv_block:  并联短路与卷积路径

# In[15]:


def identify_FC_block(X,dense_list,act_list,last_activation):
    X=Add()([identify_path(X),FC_path(X,dense_list,act_list)])
    X=Activation(last_activation)(X)
    return(X)

def identity_block(X,f,filter_list):
    X=Add()([identify_path(X),conv2d_3_path(X,f,1,filter_list)])
    X=Activation('relu')(X)
    return(X)

def convolutional_block(X,f,filter_list):
    X=Add()([conv2d_3_path(X,f,2,filter_list),conv2d_1_path(X,f,2,filter_list)])
    X=Activation('relu')(X)
    return(X)
    


# # 测试

# In[17]:


if __name__=="__main__":
    tf.reset_default_graph()

    with tf.Session() as test:
        np.random.seed(1)
        A_prev = tf.placeholder("float", [3, 4, 4, 6])
        X = np.random.randn(3, 4, 4, 6)
#         A = identity_block(A_prev, f = 2, filter_list = [2, 4, 6])
        A=identify_FC_block(A_prev,[32,32,None],['relu','relu',None],'relu')
        test.run(tf.global_variables_initializer())
        out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
        print("out = " + str(out[0][1][1][0]))


# # ResNet50
# 
# <img src="images/resnet_kiank.png" style="width:850px;height:150px;">
# <caption><center> <u> <font color='purple'> **Figure 5** </u><font color='purple'>  : **ResNet-50 model** </center></caption>
# 
# The details of this ResNet-50 model are:
# - Zero-padding pads the input with a pad of (3,3)
# - Stage 1:
#     - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
#     - BatchNorm is applied to the channels axis of the input.
#     - MaxPooling uses a (3,3) window and a (2,2) stride.
# - Stage 2:
#     - The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
#     - The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
# - Stage 3:
#     - The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
#     - The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
# - Stage 4:
#     - The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
#     - The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
# - Stage 5:
#     - The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
#     - The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
# - The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
# - The flatten doesn't have any hyperparameters or name.
# - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be `'fc' + str(classes)`.

# In[25]:


input_shape=(64, 64, 3)
X_input = Input(input_shape)
# Zero-Padding
X = ZeroPadding2D((3, 3))(X_input)
model = Model(inputs = X_input, outputs = X, name='ResNet50')

print(model)


# In[ ]:




