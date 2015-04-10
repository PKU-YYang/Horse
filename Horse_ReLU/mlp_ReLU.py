#-*- encoding:utf-8 -*-
__author__ = 'Yang'


import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from scipy.stats import itemfreq
import scipy.optimize
from theano.tensor.shared_randomstreams import RandomStreams
from logistic_cg import load_data, ConditionalLogisticRegression


def ReLU(x):
    return T.switch(x < 0, 0, x)

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

def _dropout_from_layer(rng, layer, p):
    # p 是丢掉的概率
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(89757))

    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)

    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None,
                 activation=T.tanh):
        super(DropoutHiddenLayer, self).__init__(
              rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
              activation=activation)


class MLP(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        L1_reg=0.001,
        L2_reg=0.0001,
        hidden_layers_sizes = None,
        clogit_weights_file = None,
        hiddenLayer_weights_file = None, hiddenLayer_bias_file = None,
        activation_function = T.nnet.sigmoid
    ):

        self.n_layers = len(hidden_layers_sizes)
        self.activation = activation_function

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 17)) #一个除了89757改变随机的地方

        #因为cg算法优化函数的特性，params必须放在一起
        #倒过来构建，先构建顶层conditional logit 的参数
        _params = numpy.zeros(hidden_layers_sizes[-1]+1, dtype=theano.config.floatX)
        for i in reversed(xrange(self.n_layers)):

            if i == 0: #第一个隐层
                n_in = n_ins
            else:
                n_in = hidden_layers_sizes[i-1] #输入个数是上一层的输出个数
            n_out = hidden_layers_sizes[i]
            _params = numpy.concatenate((_params, numpy.asarray(numpy_rng.uniform(
                    low = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in+1, n_out)).ravel(), dtype=theano.config.floatX)))

        self.params_length=len(_params)
        self.params = theano.shared(_params[::-1], borrow = True)
        self.ReLU_layers = []

        self.x = T.matrix('x')
        self.index = T.ivector('index')

        for i in xrange(self.n_layers):

            if i == 0:
                input_size = n_ins
                layer_input = self.x
                _nth = 0 #描述该模型的参数在整个params序列里的起始位置
            else:
                input_size = hidden_layers_sizes[i - 1] #上一层的输出
                layer_input = self.ReLU_layers[-1].output


            if hiddenLayer_weights_file is None: #如果没有读入的权重
                _nth_end =  _nth + input_size*hidden_layers_sizes[i] # W权重在params参数里的终结位置
                ReLU_layer = HiddenLayer(   rng=numpy_rng,
                                            input=layer_input, #上一层的输出
                                            n_in=input_size, #上一层的大小
                                            n_out=hidden_layers_sizes[i],
                                            W=self.params[_nth:_nth_end].reshape((input_size,hidden_layers_sizes[i])),
                                            b=self.params[_nth_end:(_nth_end+ hidden_layers_sizes[i])],
                                            activation=activation_function)
                _nth = _nth_end+hidden_layers_sizes[i] #下个模型参数开始的地方

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

                ReLU_layer = HiddenLayer( rng=numpy_rng,
                                          input=layer_input, #上一层的输出
                                          n_in=input_size, #上一层的大小
                                          n_out=hidden_layers_sizes[i],
                                          activation=activation_function,
                                          W=shared_hiddenLayer_W,
                                          b=shared_hiddenLayer_b)

            self.ReLU_layers.append(ReLU_layer)

        #最后加一层logistic，有提供权重的时候就要去读入
        if clogit_weights_file is None:

            self.clogit_Layer = ConditionalLogisticRegression(
                input=self.ReLU_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                index=self.index,
                theta=self.params[-(hidden_layers_sizes[-1]+1):] #这里也可以全部置0
            )

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

        self.L1 = abs(self.params.ravel()).sum()

        self.L2 = (self.params.ravel() ** 2).sum()

        self.cost = self.clogit_Layer.negative_log_likelihood(self.index) + L1_reg * self.L1 + L2_reg * self.L2

        self.r_error = self.clogit_Layer.Rsquare(self.index)

        self.validation_scores=[numpy.inf, 0, 0] #valid test train

        self.best_w = [None,0]

        self.best_b = [None,0]

    def build_cost_gradient_functions(self, datasets, batch_size=1500):
    #构建cg寻优算法要用的三个函数 train_fn, train_grad, callback

        train_set_x, train_set_y, train_set_index = datasets[0]
        valid_set_x, valid_set_y, valid_set_index = datasets[1]
        test_set_x, test_set_y, test_set_index = datasets[2]
        n_train_batches = (len(numpy.unique(train_set_index.eval()))-1) / batch_size

        minibatch = T.lscalar('')  # index to a [mini]batch

        ##################
        # build train_fn #
        ##################

        batch_cost = theano.function(
            [minibatch],
            self.cost, #计算在train_data上起始点开始接下来batch_size个函数的log-liklihood
            givens={
                self.x: train_set_x[train_set_index[minibatch]:train_set_index[minibatch + batch_size]],
                self.index: train_set_index[minibatch:(minibatch + batch_size + 1)] - train_set_index[minibatch]
            },
            name="batch_cost",
            allow_input_downcast=True
        )

        #根据上面提供的train set上的cost计算gradient
        batch_grad = theano.function(
            [minibatch],
            T.grad(self.cost, self.params),
            givens={
                self.x: train_set_x[train_set_index[minibatch]:train_set_index[minibatch + batch_size]],
                self.index: train_set_index[minibatch:(minibatch + batch_size + 1)] - train_set_index[minibatch]
            },
            name="batch_grad",
            allow_input_downcast=True

        )

        #调用batch_cost 在所有train set的batch上计算cost，然后输出均值
        def train_fn(params_value):
            self.params.set_value(numpy.float32(params_value), borrow=True)
            train_losses = [batch_cost(i * batch_size)
                            for i in xrange(n_train_batches)]
            return numpy.mean(train_losses)

        #调用batch_grad 在train_set的batch上累加gradient然后除以batch的个数
        def train_fn_grad(params_value):
            self.params.set_value(numpy.float32(params_value), borrow=True)
            grad = batch_grad(0)
            for i in xrange(1, n_train_batches):
                grad += batch_grad(i * batch_size)
            return grad / n_train_batches

        ##################
        # build callback #
        ##################

        # performance R2 on train set
        train_model = theano.function(
            [],
            self.r_error,
            givens={
                self.x: train_set_x,
                self.index: train_set_index
            },
            name='train',
            allow_input_downcast=True
        )

        # performance R2 on test set
        test_model = theano.function(
            [],
            self.r_error,
            givens={
                self.x: test_set_x,
                self.index: test_set_index
            },
            name='test',
            allow_input_downcast=True

        )

        #performance R2 on valid set
        validate_model = theano.function(
            [],
            self.r_error,
            givens={
                self.x: valid_set_x,
                self.index: valid_set_index
            },
            name='valid',
            allow_input_downcast=True
        )

        def callback(params_value):
            self.params.set_value(numpy.float32(params_value), borrow=True)
            this_validation_loss = validate_model()

            #为了防止overfitting，必须观测在每次validating的R2创新高的时候train是不是降低了
            this_train_loss = train_model()

            print('validation R Square %f , training R square %f ' % (1-this_validation_loss, 1-this_train_loss))

            if this_validation_loss < self.validation_scores[0]:

                self.validation_scores[0] = this_validation_loss
                self.validation_scores[2] = this_train_loss
                #因为test和valid都是一样的数据，所以先关闭这个功能
                #self.validation_scores[1] = test_model()

                #model最好的时候存权重

                self.best_w[0], self.best_b[0] = self.MLP_show_weights()


        return train_fn, train_fn_grad, callback

    def MLP_show_weights(self):

        weights=[]

        bias=[]

        for i in xrange(self.n_layers):

            weights.append(self.ReLU_layers[i].W.eval())
            bias.append(self.ReLU_layers[i].b.eval())

        #隐层的取完把conditional logit的也取出来
        weights.append(self.clogit_Layer.W.eval())
        bias.append(self.clogit_Layer.b.eval())

        return weights, bias

    def _MLP_show_hiddenlayer_output(self, test_input, i):
        #这个函数用来计算所有隐层的输出，这是一个中间函数
        #input: n*m hidden_w: m*hidden_units hidden_b:hidden_units*1
        if i==0:
            return self.activation(T.dot(test_input, self.ReLU_layers[i].W) + self.ReLU_layers[i].b)
        else:
            return self.activation(T.dot(self._MLP_show_hiddenlayer_output(test_input, i-1),
                                         self.ReLU_layers[i].W) + self.ReLU_layers[i].b)
                                        #这里的layer如果有新的读入会用新的读入

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

def train_MLP(dataset, hidden_layers, activation, weights_save, L1_reg, L2_reg,n_epochs=500, batch_size=100, optimization='BFGS'):


    datasets = load_data(dataset[0], dataset[1], dataset[2])

    train_set_x, train_set_y, train_set_index = datasets[0]

    n_in = train_set_x.shape[1].eval()  # number of features in a horse

    numpy_rng = numpy.random.RandomState(89757)

    x = T.matrix()

    print '...building the model'
    classifier = MLP(numpy_rng=numpy_rng, n_ins=n_in, hidden_layers_sizes=hidden_layers, L1_reg=L1_reg, L2_reg=L2_reg,
                     activation_function=activation)

    train_fn, train_fn_grad, callback = classifier.build_cost_gradient_functions(datasets, batch_size)

    if not os.path.exists(weights_save):
        os.makedirs(weights_save)

    os.chdir(weights_save)

    if optimization=='LBFGS':
        print ("Optimizing using scipy.optimize.fmin_l-bfgs...")
        start_time = time.clock()
        best_w_b = scipy.optimize.fmin_l_bfgs_b(
            func=train_fn,#存cost
            x0=numpy.asarray(classifier.params.get_value(), dtype=x.dtype),
            fprime=train_fn_grad, #存gradient
            callback=callback,#在train_set上每train一个minibatch后就测试在valid_set上的r2，存一个最好的，测试函数就是这里的callback
            disp=0,
            maxiter=n_epochs
            #在train data上表现最好的参数存在best_w_b[0]里面
        )

    elif optimization=='BFGS':
        print ("Optimizing using scipy.optimize.fmin_bfgs...")
        start_time = time.clock()
        best_w_b = scipy.optimize.fmin_bfgs(
            f=train_fn,#存cost
            x0=numpy.asarray(classifier.params.get_value(), dtype=x.dtype),
            fprime=train_fn_grad, #存gradient
            callback=callback,#在train_set上每train一个minibatch后就测试在valid_set上的r2，存一个最好的，测试函数就是这里的callback
            disp=0,
            maxiter=n_epochs,
            full_output=True
        )


    weights=classifier.best_w[0]
    bias=classifier.best_b[0]

    #在最外面存结果，节约时间
    numpy.savetxt("clogitLayer.csv",numpy.hstack((weights[-1].ravel(), bias[-1].ravel())),delimiter=",")

    for i in xrange(len(weights)-1):
        weights_filename=[str(i+1),"_hiddenLayer_W.csv"]
        bias_filename=[str(i+1),"_hiddenLayer_b.csv"]
        numpy.savetxt("".join(weights_filename),weights[i],delimiter=",")
        numpy.savetxt("".join(bias_filename),bias[i],delimiter=",")


    print(
        (
            'Optimization complete with best R2 on validation dataset of %f , with '
            'R2 on training dataset %f '
        )
        % (1-classifier.validation_scores[0] , 1-classifier.validation_scores[2] )
    )

    end_time = time.clock()
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)) +
                            ', with hidden layers ' + str(hidden_layers) + ', best R2 on training %f'
                            ', on validation %f ' % (1-classifier.validation_scores[2] , 1-classifier.validation_scores[0] ))
    os.chdir("../")
    #print ( 'param length: %d' % classifier.params_length )


if __name__ == '__main__':
    # train_MLP(dataset=['horse_train.csv','horse_train.csv','horse_train.csv'], hidden_layers=[256,256], weights_save="./results_2_sigmoid",
    #           n_epochs=600, batch_size=1500, activation=T.nnet.sigmoid)
    # l1=0.00001
    # for l2 in [0.0001, 0.0002,0.0003, 0.0005, 0.00008]:
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[50,50,50], weights_save="./results_2_sigmoid",
    #             n_epochs=150, batch_size=1460, L1_reg=l1, L2_reg=l2, activation=ReLU)
    #     print "l2:",l2

    # l1=0.0001
    # for l2 in [0.0001, 0.0002,0.0003, 0.0005, 0.00008]:
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[50,50,50], weights_save="./results_2_sigmoid",
    #             n_epochs=150, batch_size=1460, L1_reg=l1, L2_reg=l2, activation=ReLU)
    #     print "l2:",l2


    # l1=0.00005
    # for l2 in [0.0001, 0.0002,0.0003, 0.0005, 0.00008]:
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[50,50,50], weights_save="./results_2_sigmoid",
    #             n_epochs=150, batch_size=1460, L1_reg=l1, L2_reg=l2, activation=ReLU)
    #     print "l2:",l2


    # l1=0.000005
    # for l2 in [0.0001, 0.0002,0.0003, 0.0005, 0.00008]:
    #     train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[50,50,50], weights_save="./results_2_sigmoid",
    #             n_epochs=150, batch_size=1460, L1_reg=l1, L2_reg=l2, activation=ReLU)
    #     print "l2:",l2


    train_MLP(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], hidden_layers=[7,7], weights_save="./results_relu_sigmoid",
              n_epochs=350, batch_size=1, L1_reg=0., L2_reg=0.0000, activation=ReLU, optimization='LBFGS')