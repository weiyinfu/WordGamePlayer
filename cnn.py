import numpy as np
import tensorflow as tf

from 传统机器学习方法 import get_data, get_label, label_number, image_rows, image_cols, uni_size


def onehot(data):
    # 对data进行onehot编码
    y_cnt = len(label_number)
    y = np.zeros((len(data), y_cnt))
    for i in range(len(y)):
        y[i][data[i]] = 1
    return y


# 加载数据
x_train, y_train = get_data()
x_train = x_train.reshape(-1, image_rows, image_cols, 1)
y_train_onehot = onehot(y_train)
# x_train = x_train[:10]
# y_train_onehot = y_train_onehot[:10]

# 学习率等神经网络超参数
num_output = len(label_number)
learning_rate = 0.001
display_epoch = 10
num_epoch = 1000
keep_prop = 0.5
model_path = "model/cnn"
"""
kernel_size=5,7
pool卷积核为2时效果比较好
"""


def conv(cin, cout, stride=1, kernel_size=7):
    # 卷积层
    w = tf.Variable(tf.random_normal([kernel_size, kernel_size, int(cin.shape[-1]), cout]))
    b = tf.Variable(tf.random_normal([cout]))
    output = tf.nn.conv2d(cin, w, [1, stride, stride, 1], padding='SAME')
    output = tf.nn.relu(tf.nn.bias_add(output, b))
    return output


def pool(cin, kernel_size=2):
    # 池化层
    return tf.nn.max_pool(cin, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, kernel_size, kernel_size, 1],
                          padding='SAME')


def fcn(cin, cout, keep_prop, is_last_layer=False):
    # 全连接层
    w = tf.Variable(tf.random_normal([cin.shape[1].value, cout]))
    b = tf.Variable(tf.random_normal([cout]))
    output = tf.add(tf.matmul(cin, w), b)
    if is_last_layer:
        output = tf.nn.relu(output)
        output = tf.nn.dropout(output, keep_prop)
    return output


# 网络结构
x_place = tf.placeholder(tf.float32, (None, image_rows, image_cols, 1), name="x_place")
y_place = tf.placeholder(tf.float32, (None, num_output), name="y_place")
keep_prob_place = tf.placeholder(tf.float32, name="keep_prop")
fc1 = pool(conv(pool(conv(x_place, 32)), 64))
fc1 = tf.reshape(fc1, (-1, np.prod(fc1.shape[1:])))
fc2 = fcn(fc1, 1024, keep_prop=keep_prob_place)
logits = fcn(fc2, num_output, True)

# 梯度和精度
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_place, logits=logits))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
y_mine = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_mine, tf.argmax(y_place, axis=1)), tf.float32))


def train():
    # 训练过程
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epoch):
            _, lo = sess.run([train_op, loss], feed_dict={
                x_place: x_train,
                y_place: y_train_onehot,
                keep_prob_place: keep_prop
            })
            print(epoch, 'loss', lo)
            if lo == 0:
                break
        saver.save(sess, model_path)


def test():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        print(sess.run(accuracy, feed_dict={
            x_place: x_train,
            y_place: y_train_onehot,
            keep_prob_place: 1.0
        }))


class CNN:
    def __init__(self):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)

    def predict(self, recs):
        x = uni_size([i['data'] for i in recs])
        x = x.reshape(-1, image_rows, image_cols, 1)
        myans = self.sess.run(y_mine, feed_dict={
            x_place: x,
            keep_prob_place: 1.0
        })
        print(recs)
        for i in range(len(recs)):
            recs[i]['ans'] = get_label(myans[i])
        return recs


if __name__ == '__main__':
    train()
    # test()
