import keras

from model import mobilenet_v1
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import optimizers
from data import train_iterator


def train_step(model, images, labels, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        # print(prediction[0])
        # print(sum(obj))

        preobj = prediction[:,:,:,:2]
        # prexoft = tf.reshape(prediction[:,:,:,1:2],(-1,1))
        # preyoft = tf.reshape(prediction[:,:,:,2:3],(-1,1))

        gtobj = labels[:,:,:,:2]
        weight = tf.constant([[[[1,100]]]],dtype=tf.float32)
        weight = tf.reduce_sum(weight*gtobj,axis=3)

        # print(weight)
        gtxoft = tf.reshape(labels[:,:,:,1:2],(-1,1))
        gtyoft = tf.reshape(labels[:,:,:,2:3],(-1,1))


        loss1 =tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=gtobj,logits=preobj),weight)
        loss1 = tf.reduce_mean(loss1)
        # print(loss1)

        # mse = tf.keras.losses.MeanSquaredError()
        # loss2 = mse(prexoft,gtxoft)*tf.reshape(gtobj,(-1,1))
        # loss2 = tf.reduce_sum(loss2)
        # # print(loss2)
        #
        # loss3 = mse(preyoft,gtyoft)*tf.reshape(gtobj,(-1,1))
        # loss3 = tf.reduce_sum(loss3)
        # print(loss3)
        loss = loss1#+loss2+loss3
        # loss /= 30
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, prediction

def train(model, data_iterator, optimizer):

    for i in tqdm(range(int(143/ 30))):
        images, labels = data_iterator.next()
        ce, prediction = train_step(model, images, labels, optimizer)

        print('ce: {:.4f}'.format(ce))

class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)
    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return step / self.warm_up_step * self.initial_learning_rate
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)

if __name__ == '__main__':
    train_data_iterator = train_iterator()
    model = mobilenet_v1(input_shape=[128, 128, 3],  # 模型输入图像shape
                      alpha=0.25,  # 超参数，控制卷积核个数
                      depth_multiplier=1,  # 超参数，控制图像分辨率
                      dropout_rate=1e-3)  # 随即杀死神经元的概率

    model.build(input_shape=(None,) + (128,128,3))

    # model = tf.keras.models.load_model("./beizi16m.h5")
    # model.summary()

    # optimizer = optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    # optimizer = optimizers.Adam()

    import tensorflow_addons as tfa
    optimizers = [
        tf.keras.optimizers.Adam(learning_rate=1e-4),
        tf.keras.optimizers.Adam()
    ]
    optimizers_and_layers = [(optimizers[0], model.layers[:-5]), (optimizers[1], model.layers[-5:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    for epoch_num in range(200):
        train(model, train_data_iterator, optimizer)
        model.save('./beizi64m.h5', save_format='h5')

