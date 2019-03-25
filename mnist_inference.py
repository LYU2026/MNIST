# -*- coding: UTF-8 -*-

import tensorflow as tf

# 定义神经网络结构的相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 通过tf.get_variable 函数获取变量。
# 在训练神经网络时会创建这些变量；在测试时会通过保存的模型加载这些变量的取值
# 将变量的正则化损失加入损失集合
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weight", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    # 将当前变量的正则化损失加入自定义集合losses
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))  # tf.add_to_collections?
    return weights

# 定义神经网络的前向传播过程
def inference(input_tensor,regularizer):
    # 声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable(
            "biases",[LAYER1_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)

    # 声明第二层神经网络
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable(
            "biases",[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer2=tf.matmul(layer1,weights)+biases

    return layer2