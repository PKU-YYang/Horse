#-*- encoding:utf-8 -*-
__author__ = 'Yang'
# 相比于mlp_relu 多dropout+momentuem+SGD

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from scipy.stats import itemfreq
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_cg import load_data
from collections import OrderedDict
from theano.ifelse import ifelse


def ReLU(x):
    return T.switch(x < 0, 0, x)

class ConditionalLogisticRegression(object):

    def __init__(self, input, n_in, index, theta=None, W=None, b=None): #input是一个minibatch，单位是一组赛事，不是一个sample


        n_out=1  #对于CL模型来说，并不是每一类构建一个分类平面，一直都只有一个数值,就是每匹马夺冠的概率

        #把W和b写在theta里面方便T.grad

        if theta is None:
            self.theta = theano.shared(
                value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
                ),
                name='theta',
                borrow=True
            )
        else:
            self.theta = theta

        _W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        _b = self.theta[n_in * n_out:n_in * n_out + n_out]

        if W is None:
            self.W = _W
            self.b = _b
        else:
            self.W = W
            self.b = b

        # 把线性回归的值exp之后再按组归一化就是最后的值
        _raw_w = T.exp(T.dot(input, self.W) + self.b)

        # 计算每组比赛内的exp和
        def cumsum_within_group(_start, _index, _race):
            start_point = _index[_start]
            stop_point = _index[_start+1]
            return T.sum(_race[start_point:stop_point], dtype='float32')

        # _cumsum就是每组的exp的合
        _cumsum, _ = theano.scan(cumsum_within_group,
                                 sequences=[T.arange(index.shape[0]-1)],
                                 non_sequences=[index, _raw_w])

        # index实际的样子是从0开始每一位记录每组赛事的第一名的位置，也就是每组比赛开始的地方，最后一位是全部输入sample的行数
        # index在读入的时候被默认做了一步转化，从 [1,1,1,1,2,2,2] 到 [0,4,7]
        # 构造一个rep(cumsum,times)的序列，目的是直接相除从而得到每匹马的概率
        # _times里存的是每组比赛的马的数量
        self._times, _ = theano.scan(fn=lambda i, index: index[i+1]-index[i],
                                     sequences=[T.arange(index.shape[0]-1)],
                                     non_sequences=index)

        _raceprobdiv = T.ones_like(_raw_w)

        # 这里运用的技巧是构造一个等长的序列，然后用T.set_subtensor改变里面的值，SCAN不允许每次输出长度不一样的序列，所以不可以concatenate
        def change_race_prob_div(_i, _change, _rep, _times, _item):
            _change = T.set_subtensor(_change[_rep[_i]:_rep[_i+1]], T.reshape(T.alloc(_item[_i],_times[_i]),(_times[_i],1)))
            return _change

        # _race_prob_div存的是每一位对应的要除的概率归一化的值
        _race_prob_div, _ = theano.scan(fn = change_race_prob_div,
                                        sequences=[T.arange(index.shape[0]-1)],
                                        outputs_info=[_raceprobdiv],
                                        non_sequences=[index,self._times, _cumsum])

        # 归一化以后的概率值,整个init过程最重要的就是计算每匹马的概率，在普通的logistic里计算这个不需要label,只要一个softmax就行
        self.race_prob = _raw_w / _race_prob_div[-1]

        self.mean_neg_loglikelihood = None

        self.neg_log_likelihood = None

        self.pos_log_likelihood=None

        self.r_square = None

        self.r_error = None

        self.params = [self.theta] # clogit的参数theta

    def negative_log_likelihood(self, index):

        #特别注意：output_info一定不能用numpy组成的序列，用也要禁掉broadcast，或者干脆用shared variable
        _output_info = T.as_tensor_variable(numpy.array([0.]))

        _output_info = T.unbroadcast(_output_info, 0)

        # _1st_prob存的是对每次比赛第一匹马的likelihood求和的过程
        _1st_prob, _ = theano.scan(fn= lambda _1st, prior_reuslt, _prob: prior_reuslt+T.log(_prob[_1st]),
                                   sequences=[index[:-1]],
                                   outputs_info=_output_info,
                                   non_sequences=self.race_prob)

        self.neg_log_likelihood = 0. - _1st_prob[-1] #这个是负的

        self.pos_log_likelihood = _1st_prob[-1]

        self.mean_neg_loglikelihood = self.neg_log_likelihood/(index.shape[0]-1)

        #因为cost必须是0维的，所以用T.mean巧妙的转换一下
        return T.mean(self.mean_neg_loglikelihood.ravel(), dtype='float32')

    def Rsquare(self, index): #rsqaure约大越好，函数返回的值越小越好

        _output_info = T.as_tensor_variable(numpy.array([0.]))

        _output_info = T.unbroadcast(_output_info, 0)

        # rsquare计算是除以Ln(1/n_i),n_i是每组比赛中马的个数
        _r_square_div, _ = theano.scan(fn = lambda _t, prior_reuslt: prior_reuslt+T.log(1./_t),
                                       sequences=[self._times],
                                       outputs_info=_output_info #特别注意：output_info一定不能用numpy组成的序列，用shared或者禁掉broadcast
                                       )

        self.r_error = self.pos_log_likelihood / _r_square_div[-1]

        self.r_square = 1 - self.r_error

        #用T.mean转化成一维的
        return T.mean(self.r_error.ravel(), dtype='float32')

    def show_theta(self):

        return self.theta.get_value()

class HiddenLayer(object):

    # 一个隐层，初始化W,b,并且告诉输入有输出
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)


        self.W = W
        self.b = b
        # hidden layer的输出是连续数值，在0,1之间
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

# 把一个layer的输出drop out
def _dropout_from_layer(rng, layer, p):
    # p 是丢掉的概率
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(89757))

    mask = srng.binomial(n=1, p=1-p, size=layer.shape)

    output = layer * T.cast(mask, theano.config.floatX)
    return output

# 和普通的hiddenlayer一样构造一个dropout layer，然后按比例扔掉它的output
class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None,
                 activation=T.tanh, dropout_rate=None):
        super(DropoutHiddenLayer, self).__init__(
              rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
              activation=activation)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

class MLP(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=None,
        input=None,
        index=None,
        dropout_rates=None,
        clogit_weights_file=None,
        hiddenLayer_weights_file=None, hiddenLayer_bias_file=None,
        activation_function=T.nnet.sigmoid
    ):

        self.n_layers = len(hidden_layers_sizes)
        self.activation = activation_function

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 17)) #一个除了89757改变随机的地方

        self.ReLU_layers = []    # 非train时用来存放W乘以0.5的输出
        self.dropout_layers = [] # train时用来存放输出被drop out的layer

        # 先dropout第一层,也就是输入层
        next_dropout_layer_input = _dropout_from_layer(numpy_rng, input, p=dropout_rates[0])
        next_layer_input = input

        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1] # 上一层的输出

            if hiddenLayer_weights_file is None: # 如果没有读入的权重

                # train的时候，这一层的输入是上一层dropout的输出，然后这层输出dropout给下一层作为输入
                this_dropout_layer = DropoutHiddenLayer(rng=numpy_rng, input=next_dropout_layer_input,
                                                        n_in=input_size, n_out=hidden_layers_sizes[i],
                                                        activation=activation_function,
                                                        dropout_rate=dropout_rates[i+1]) # 输入层的dropout已经用掉了
                self.dropout_layers.append(this_dropout_layer)
                next_dropout_layer_input = this_dropout_layer.output

                # test 的时候把输出的权重*(1-p)
                this_ReLU_layer = HiddenLayer(rng=numpy_rng,
                                              input=next_layer_input, # 上一层的输出
                                              n_in=input_size, # 上一层的大小
                                              n_out=hidden_layers_sizes[i],
                                              W=this_dropout_layer.W * (1 - dropout_rates[i]),
                                              # drop rate是错位的，因为w*x, x用的是上一层的dropout rate
                                              b=this_dropout_layer.b,
                                              activation=activation_function)
                self.ReLU_layers.append(this_ReLU_layer)
                next_layer_input = this_ReLU_layer.output

            else:

                print "Reading in the weights and bias of %d Hidden Layer" % (i+1)

                weights_filename = "".join([str(i+1), "_hiddenLayer_W.csv"])
                bias_filename = "".join([str(i+1), "_hiddenLayer_b.csv"])

                f = open(os.path.join(hiddenLayer_weights_file, weights_filename), "rb")
                data = numpy.loadtxt(f, delimiter=',', dtype=float)
                f.close()
                shared_hiddenLayer_W = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)

                f = open(os.path.join(hiddenLayer_bias_file, bias_filename), "rb")
                data = numpy.loadtxt(f, delimiter=',', dtype=float)
                f.close()
                shared_hiddenLayer_b = theano.shared(numpy.asarray(data, dtype=theano.config.floatX), borrow=True)

                this_ReLU_layer = HiddenLayer( rng=numpy_rng,
                                               input=next_layer_input, # 上一层的输出
                                               n_in=input_size, # 上一层的大小
                                               n_out=hidden_layers_sizes[i],
                                               activation=activation_function,
                                               W=shared_hiddenLayer_W,
                                               b=shared_hiddenLayer_b)

                self.ReLU_layers.append(this_ReLU_layer)
                next_layer_input = this_ReLU_layer.output

        # clogit，有提供权重的时候就要去读入
        if clogit_weights_file is None:

            # train的时候的最后一层
            dropout_clogit_Layer = ConditionalLogisticRegression(
                input=next_dropout_layer_input,
                n_in=hidden_layers_sizes[-1],
                index=index
            ) # 最后一层的输出不需要dropout
            self.dropout_layers.append(dropout_clogit_Layer)

            # test时候的最后一层，W是根据上一层的dropout做变换
            clogit_Layer = ConditionalLogisticRegression(
                input=next_layer_input,
                n_in=hidden_layers_sizes[-1],
                index=index,
                W=dropout_clogit_Layer.W * (1 - dropout_rates[-1]),
                b=dropout_clogit_Layer.b
            )
            self.ReLU_layers.append(clogit_Layer)
        else:

            print "Reading in the weights and bias of Conditional Logit Layer \n"

            #把conditional logit layer的权重读入
            weights_filename=os.path.join(clogit_weights_file, "clogitLayer.csv")

            f = open(weights_filename,"rb")
            data = numpy.loadtxt(f, delimiter=',', dtype=float)
            f.close()
            shared_clogit_theta = theano.shared(numpy.asarray(data[0:n_ins], dtype=theano.config.floatX), borrow=True)

            self.clogit_Layer = ConditionalLogisticRegression(
                input=self.ReLU_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                index=self.index,
                theta=shared_clogit_theta
            )

        # train的时候用dropout layer, 正常使用的参数是RELU layer的
        self.dropout_cost = self.dropout_layers[-1].negative_log_likelihood(index)
        self.dropout_r_error = self.dropout_layers[-1].Rsquare(index)

        self.cost = self.ReLU_layers[-1].negative_log_likelihood(index)
        self.r_error = self.ReLU_layers[-1].Rsquare(index)

        self.best_w = [None, 0] # 用来存放最好模型的参数
        self.best_b = [None, 0]

        # train的时候用的是dropout layer
        self.params = [param for layer in self.dropout_layers for param in layer.params]

    def MLP_show_weights(self):

        weights = []
        bias = []

        for i in xrange(self.n_layers):
            weights.append(self.ReLU_layers[i].W.eval())
            bias.append(self.ReLU_layers[i].b.eval())

        # 隐层的取完把conditional logit的也取出来
        weights.append(self.ReLU_layers[-1].W.eval())
        bias.append(self.ReLU_layers[-1].b.eval())

        return weights, bias

    # 如果要额外读取test data，那要把这里的W修改为 W * drop rate
    def _MLP_show_hiddenlayer_output(self, test_input, i):
        # 这个函数用来计算所有隐层的输出，这是一个中间函数
        # input: n*m hidden_w: m*hidden_units hidden_b:hidden_units*1
        if i==0:
            return self.activation(T.dot(test_input, self.ReLU_layers[i].W) + self.ReLU_layers[i].b)
        else:
            return self.activation(T.dot(self._MLP_show_hiddenlayer_output(test_input, i-1),
                                         self.ReLU_layers[i].W) + self.ReLU_layers[i].b)
                                        # 这里的layer如果有新的读入会用新的读入

    def MLP_test_output(self, test_input, test_index):

        _test_raw_w = T.exp(T.dot(self._MLP_show_hiddenlayer_output(test_input, self.n_layers-1),
                                  self.clogit_Layer.W) + self.clogit_Layer.b)

        def cumsum_within_group(_start, _index, _race):
            start_point = _index[_start]
            stop_point = _index[_start+1]
            return T.sum(_race[start_point:stop_point], dtype='float32')

        # _cumsum就是每组的exp的合
        _cumsum, _ = theano.scan(cumsum_within_group,
                                 sequences=[T.arange(test_index.shape[0]-1)],
                                 non_sequences=[test_index, _test_raw_w])

        # 看每个cumsum重复了多少次
        _times, _ = theano.scan(fn=lambda i, index: index[i+1]-index[i],
                                sequences=[T.arange(test_index.shape[0]-1)],
                                non_sequences=test_index)

        _raceprobdiv = T.ones_like(_test_raw_w)

        def change_race_prob_div(_i, _change, _rep, _times, _item):
            _change = T.set_subtensor(_change[_rep[_i]:_rep[_i+1]], T.reshape(T.alloc(_item[_i],_times[_i]),(_times[_i],1)))
            return _change

        # _race_prob_div存的是每一位对应的要除的概率归一化的值
        _race_prob_div, _ = theano.scan(fn = change_race_prob_div,
                                        sequences=[T.arange(test_index.shape[0]-1)],
                                        outputs_info=[_raceprobdiv],
                                        non_sequences=[test_index,_times, _cumsum])

        #归一化以后的概率值
        _test_race_prob = _test_raw_w / _race_prob_div[-1]

        return _test_race_prob

def train_MLP(initial_learning_rate,
              learning_rate_decay,
              mom_params,
              dropout,
              dropout_rates,
              dataset,
              hidden_layers, activation,
              weights_save,
              n_epochs=500,
              batch_size=100,
              ):

    # dropout rate多一个输入层的值
    assert len(hidden_layers) + 1 == len(dropout_rates)

    datasets = load_data(dataset[0], dataset[1], dataset[2])
    train_set_x, train_set_y, train_set_index = datasets[0]
    valid_set_x, valid_set_y, valid_set_index = datasets[1]

    n_train_batches = (len(numpy.unique(train_set_index.eval()))-1) / batch_size
    n_in = train_set_x.shape[1].eval()  # number of features in a horse

    # params
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    learning_rate = theano.shared(numpy.asarray(initial_learning_rate, dtype=theano.config.floatX))
    numpy_rng = numpy.random.RandomState(89757)

    x = T.matrix()
    index = T.ivector()
    minibatch = T.lscalar()  # index to a [mini]batch
    epoch = T.scalar() # epoch 决定momentuem

    print '...building the model'
    classifier = MLP(numpy_rng=numpy_rng, n_ins=n_in, hidden_layers_sizes=hidden_layers,
                     dropout_rates=dropout_rates,
                     input=x, index=index,
                     activation_function=activation)

    cost = classifier.cost
    dropout_cost = classifier.dropout_cost

    # functions to output performance on train/test/valid
    train_perform = theano.function(
        [],
        classifier.r_error,
        givens={
            x: train_set_x,
            index: train_set_index
        },
        name='train',
        allow_input_downcast=True
    )

    valid_perform = theano.function(
        [],
        classifier.r_error,
        givens={
            x: valid_set_x,
            index: valid_set_index
        },
        name='valid',
        allow_input_downcast=True
    )
    # theano.printing.pydotprint(validate_model, outfile="validate_file.png",
    #        var_with_name_simple=True)

    # compute the ordinary gradients
    gparams = []
    for param in classifier.params:
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    # place to save the special gradient by momentum
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                                   dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # dynamic momentum rate, 每个epoch不一样
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom = ifelse(epoch < mom_epoch_interval,
          mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
          mom_end)

    # update the step direction using momen
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam

    # take a step along the direction
    for param, gparam_mom in zip(classifier.params, gparams_mom):
        stepped_param = param + updates[gparam_mom] # 这里还可以做一步weight decay
        updates[param] = stepped_param


    # compile train function
    output = dropout_cost if dropout else cost
    train_model = theano.function(
            inputs=[epoch, minibatch], # epoch 决定Mom rate
            outputs=output,
            updates=updates,
            givens={
                x: train_set_x[train_set_index[minibatch]:train_set_index[minibatch + batch_size]],
                index: train_set_index[minibatch:(minibatch + batch_size + 1)] - train_set_index[minibatch]
            })
    # theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # dynamic learning_rate
    decay_learning_rate = theano.function(
                            inputs=[],
                            outputs=learning_rate,
                            updates={learning_rate: learning_rate * learning_rate_decay})

    print '... start training'

    best_valid_error = 0
    its_train_error = 0

    epoch_counter = 0
    start_time = time.clock()

    while epoch_counter < n_epochs:

        epoch_counter = epoch_counter + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(epoch_counter, minibatch_index)

        this_train_error = 1.0 - train_perform() # r2 越大越好
        this_valid_error = 1.0 - valid_perform()

        print ("epoch %d, train R2 %.6f, test R2 %.6f, learning rate= %.4f" %(
                epoch_counter, this_train_error, this_valid_error,
                learning_rate.get_value(borrow=True)) +
                (" ***" if this_valid_error > best_valid_error else ""))

        if this_valid_error > best_valid_error:

            classifier.best_w[0], classifier.best_b[0] = classifier.MLP_show_weights()
            best_valid_error = this_valid_error
            its_train_error = this_train_error
            print >> sys.stderr, ('epoch %d' % epoch_counter + ', best R2 on training %f'
                                  ', on validation %f ' % (its_train_error , best_valid_error))

        new_learning_rate = decay_learning_rate()

    # save the best weights
    if not os.path.exists(weights_save):
        os.makedirs(weights_save)
    os.chdir(weights_save)

    weights = classifier.best_w[0]
    bias = classifier.best_b[0]

    # 在最外面存结果，节约时间
    numpy.savetxt("clogitLayer.csv",numpy.hstack((weights[-1].ravel(), bias[-1].ravel())),delimiter=",")

    for i in xrange(len(weights)-1):
        weights_filename=[str(i+1), "_hiddenLayer_W.csv"]
        bias_filename=[str(i+1), "_hiddenLayer_b.csv"]
        numpy.savetxt("".join(weights_filename), weights[i], delimiter=",")
        numpy.savetxt("".join(bias_filename), bias[i], delimiter=",")

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)) +
                            ', with hidden layers ' + str(hidden_layers) + ', best R2 on training %f'
                            ', on validation %f ' % (its_train_error , best_valid_error))
    os.chdir("../")


if __name__ == '__main__':

    # 这个cl模型要求输入的比赛第一匹马必须是冠军

    # if sys.argv[1] == '1':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[128,128,128],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '2':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[256,256,256],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '3':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[512,512,512],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    #
    # if sys.argv[1] == '4':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024,1024,1024],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '5':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[128,128,128,128],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5, 0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '6':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[256]*4,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5, 0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '7':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[512]*4,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '8':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024]*4,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '9':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024]*5,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5,0.5,0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9995,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '10':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024]*6,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=ReLU,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5,0.5,0.5,0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9995,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '11':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[128,128,128],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '12':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[256,256,256],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '13':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[512,512,512],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    #
    # if sys.argv[1] == '14':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024,1024,1024],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5],
    #               initial_learning_rate=1.5, learning_rate_decay=0.9988,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '15':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[128,128,128,128],
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5, 0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '16':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[256]*4,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5, 0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '17':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[512]*4,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '18':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024]*4,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5, 0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9987,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '19':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024]*5,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5,0.5,0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9995,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})
    #
    # if sys.argv[1] == '20':
    #
    #     weights_save = "./results_"
    #     _save = "".join([weights_save, sys.argv[1]])
    #
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[1024]*6,
    #               weights_save=_save,
    #               n_epochs=3000, batch_size=3300, activation=T.nnet.sigmoid,
    #               dropout=True, dropout_rates=[0.1, 0.5, 0.5,0.5,0.5,0.5,0.5],
    #               initial_learning_rate=2., learning_rate_decay=0.9995,
    #               mom_params={"start": 0.5,
    #                           "end": 0.99,
    #                           "interval": 1400})


    train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[50,50,50],
              weights_save="./results_",
              n_epochs=1000, batch_size=3300, activation=ReLU,
              dropout=True, dropout_rates=[0.1,0.5,0.5,0.5],
              initial_learning_rate=.1, learning_rate_decay=0.995,
              mom_params={"start": 0.5,
                          "end": 0.95,
                          "interval": 500})