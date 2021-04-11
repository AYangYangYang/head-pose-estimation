import os
import sys
sys.path.append('..')
import logging
import argparse
import pandas as pd
import numpy as np

from lib.FSANET_model import *
from lib.SSRNET_model import *

import TYY_callbacks
from TYY_generators import *

from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

logging.basicConfig(level=logging.DEBUG)

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["image"], d["pose"]

def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for head pose estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=90,
                        help="number of epochs")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="validation split ratio")
    parser.add_argument("--model_type", type=int, default=3,
                        help="type of model")
    parser.add_argument("--db_name", type=str, default='300W_LP',
                        help="type of model")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    db_name = args.db_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    validation_split = args.validation_split
    model_type = args.model_type
    image_size = 64

    logging.debug("Loading data...")

    if db_name == '300W_LP':
        # 获得对应的npz文件
        db_list = ['AFW.npz','AFW_Flip.npz','HELEN.npz','HELEN_Flip.npz','IBUG.npz','IBUG_Flip.npz','LFPW.npz','LFPW_Flip.npz']
        # 用于保存像素
        image = []
        # 用于保存姿态
        pose = []
        # 循环加入所有的图片像素，以及对应的姿态
        for i in range(0,len(db_list)):
            image_temp, pose_temp = load_data_npz('../data/type1/'+db_list[i])
            image.append(image_temp)
            pose.append(pose_temp)
        # 把链表转化为np数组格式.
        # 加载完数据之后为[122450, 64, 64, 3]
        image = np.concatenate(image,0)
        # 加载完数据之后为[122450, 3]
        pose = np.concatenate(pose,0)

        # 对于其角度不在[-99,99]之间的数据，全部剔除掉
        x_data = []
        y_data = []
        print(image.shape)
        print(pose.shape)
        for i in range(0,pose.shape[0]):
            temp_pose = pose[i,:]
            if np.max(temp_pose)<=99.0 and np.min(temp_pose)>=-99.0:
                x_data.append(image[i,:,:,:])
                y_data.append(pose[i,:])
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        print(x_data.shape)
        print(y_data.shape)
    elif db_name == 'synhead_noBIWI':
        image, pose = load_data_npz('../data/synhead/media/jinweig/Data2/synhead2_release/synhead_noBIWI.npz')
        x_data = image
        y_data = pose
    elif db_name == 'BIWI':
        image, pose = load_data_npz('../data/BIWI_train.npz')
        x_train = image
        y_train = pose
        image_test, pose_test = load_data_npz('../data/BIWI_test.npz')
        x_test = image_test
        y_test = pose_test
    else:
        print('db_name is wrong!!!')
        return

    # 训练到30批次和60批次会进行学习率衰减
    start_decay_epoch = [30,60]
    # 优化器
    optMethod = Adam()
    # 论文中Stage的数目
    stage_num = [3,3,3]
    lambda_d = 1
    # 输出姿态为yaw, pitch, roll
    num_classes = 3
    isFine = False

    # 根据model_type参数 进行模型构建
    if model_type == 0:
        model = SSR_net_ori_MT(image_size, num_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_ori_mt'

    elif model_type == 1:
        model = SSR_net_MT(image_size, num_classes, stage_num, lambda_d)()
        save_name = 'ssrnet_mt'

    elif model_type == 2:
        num_capsule = 3
        dim_capsule = 16
        # steam数量
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_capsule'+str_S_set
    
    elif model_type == 3:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_capsule'+str_S_set

    elif model_type == 4:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_capsule'+str_S_set

    elif model_type == 5:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_netvlad'+str_S_set

    elif model_type == 6:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_netvlad'+str_S_set
    
    elif model_type == 7:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_netvlad'+str_S_set

    elif model_type == 8:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_metric'+str_S_set

    elif model_type == 9:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_Var_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_var_metric'+str_S_set
    elif model_type == 10:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
        str_S_set = ''.join('_'+str(x) for x in S_set)

        model = FSA_net_noS_Metric(image_size, num_classes, stage_num, lambda_d, S_set)()
        save_name = 'fsanet_noS_metric'+str_S_set

    # 指定模型的优化方法，以及loss（均值绝对误差）计算方式，
    model.compile(optimizer=optMethod, loss=["mae"],loss_weights=[1])

    logging.debug("Model summary...")
    # 计算模型参数，打印模型结构
    model.count_params()
    model.summary()

    logging.debug("Saving model...")
    # 创建必要的目录，如保存模型的路径等等
    mk_dir(db_name+"_models")
    mk_dir(db_name+"_models/"+save_name)
    mk_dir(db_name+"_checkpoints")
    # 把模型绘画成图，便于分析（总体结构）
    plot_model(model, to_file=db_name+"_models/"+save_name+"/"+save_name+".png")
    # 绘画网络模型的细致结构
    for i_L,layer in enumerate(model.layers):
        if i_L >0 and i_L< len(model.layers)-1:
            if 'pred' not in layer.name and 'caps' != layer.name and 'merge' not in layer.name and 'model' in layer.name:
                plot_model(layer, to_file=db_name+"_models/"+save_name+"/"+layer.name+".png")

    # 迭代到指定次数，进行学习率衰减
    decaylearningrate = TYY_callbacks.DecayLearningRate(start_decay_epoch)
    # 查看指定路径下的模型是否存在，存在则自动加载该目录下的模型
    callbacks = [ModelCheckpoint(db_name+"_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"), decaylearningrate
                        ]

    logging.debug("Running training...")

    # 如果为'BIWI'数据集,则进行测试集和训练集的划分
    if db_name != 'BIWI':
        data_num = len(x_data)
        indexes = np.arange(data_num)
        np.random.shuffle(indexes)
        x_data = x_data[indexes]
        y_data = y_data[indexes]
        train_num = int(data_num * (1 - validation_split))
        
        x_train = x_data[:train_num]
        x_test = x_data[train_num:]
        y_train = y_data[:train_num]
        y_test = y_data[train_num:]
    elif db_name == 'BIWI':
        train_num = np.shape(x_train)[0]

    # 为模型绑定训练数据，测试数据，并且进行训练
    hist = model.fit_generator(generator=data_generator_pose(X=x_train, Y=y_train, batch_size=batch_size),
                                       steps_per_epoch=train_num // batch_size,
                                       validation_data=(x_test, y_test),
                                       epochs=nb_epochs, verbose=1,
                                       callbacks=callbacks)
    
    logging.debug("Saving weights...")
    model.save_weights(os.path.join(db_name+"_models/"+save_name, save_name+'.h5'), overwrite=True)
    pd.DataFrame(hist.history).to_hdf(os.path.join(db_name+"_models/"+save_name, 'history_'+save_name+'.h5'), "history")


if __name__ == '__main__':
    main()