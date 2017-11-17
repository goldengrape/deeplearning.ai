
# coding: utf-8

# # 并联网络
# 
# 仿照ResNet, 建立并联网络. 
# 
# 其中并联网络可以是: 
# * 主路径为全连接网络, 捷径为identity或全连接网络
# * 主路径为Conv2d网络, 捷径为identiy或Conv2d网络
# * >=2条路径
# * 必要时并联网络可以级联
# 
# 本程序中大量参考deeplearning.ai第4门课中第2周的作业ResNet

# In[124]:


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
# 仿照并联电路的设定

# In[133]:


def parallel_connection(X,pathNet_list,activation):
    X_branch=[pathNet(X) for pathNet in pathNet_list]
    for each_branch in X_branch:
        assert X.get_shape().as_list()[1:]==each_branch.get_shape().as_list()[1:]
    X=Add()(X_branch)
    X=Activation(activation)(X)
    return X


# ## 并联路径示意

# In[134]:


def identify_path(X):
    return X

def FC_path(X):
    dense_list=[32,32]
    activation_list=['relu','relu']
    
    def fc(X,dense_list,activation_list):
        # 获取原shape
        origin_shape=X.get_shape().as_list()[1:]
        
        # 展平主路径, 如果不是m,n_H,n_W,n_C形式的, 则不需要使用Flatten
        if len(origin_shape)>=3:
            X = Flatten()(X) # Flatten

        # 主路径添加全连接层
        for stage in range(len(dense_list)):
            d=dense_list[stage]
            act=activation_list[stage]
            X = Dense(d, 
                  activation=act, name='fc_main_' + str(stage), 
                  kernel_initializer = glorot_uniform(seed=0))(X)
        # 合并前一层, 维度重整, 无激活函数
        X = Dense(np.prod(origin_shape),
                  activation=None, name='fc_main_' + str(stage), 
                  kernel_initializer = glorot_uniform(seed=0))(X)
        X = Reshape(origin_shape)(X)
        return X
    X=fc(X,dense_list,activation_list)
    return X
    


# In[135]:


def identify_FC_block(X):
    pathNet_list=[identify_path,FC_path]
    activation='relu'
    X = parallel_connection(X,pathNet_list,activation)
    return(X)


# In[136]:


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4,4 ,6])
    X = np.random.randn(3,4,4 ,6)
    dense_list=[32]
    activation_list=["relu"]
    A = identify_FC_block(A_prev)
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
#     print("out = " + str(out[0][1][1][0]))
    print(out)


# In[ ]:





# In[ ]:




