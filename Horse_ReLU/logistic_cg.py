#-*- encoding:utf-8 -*-

__docformat__ = 'restructedtext en'


import os
import sys
import time
import numpy
import theano
import theano.tensor as T



def load_data(trainset, validset, testset):

    #分别读入三个文件并share他们
    data=numpy.loadtxt(trainset, delimiter=',', dtype=float, skiprows=1)
    train_set=(data[:,:-2],data[:,-2],data[:,-1])

    data = numpy.loadtxt(validset, delimiter=',', dtype=float, skiprows=1)
    valid_set=(data[:,:-2],data[:,-2],data[:,-1]) #feature,label,raceid

    data=numpy.loadtxt(testset, delimiter=',', dtype=float, skiprows=1)
    test_set=(data[:,:-2],data[:,-2],data[:,-1])

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y, data_index = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_index = theano.shared(numpy.asarray(data_index,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32'), T.cast(shared_index, 'int32')

    test_set_x, test_set_y, test_set_index = shared_dataset(test_set)
    valid_set_x, valid_set_y, valid_set_index = shared_dataset(valid_set)
    train_set_x, train_set_y, train_set_index = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y, train_set_index), (valid_set_x, valid_set_y, valid_set_index),
            (test_set_x, test_set_y, test_set_index)]
    return rval


class ConditionalLogisticRegression(object):

    def __init__(self, input, n_in, index): #input是一个minibatch


        n_out=1 #对于CL模型来说，并不是每一类构建一个分类平面，一直都只有一个数值

        self.theta = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]

        # 计算概率的exp归一化矩阵,group by index
        _raw_w = T.exp(T.dot(input, self.W) + self.b)

        #计算每组比赛内的和
        def cumsum_within_group(_start, _index, _race):
            start_point=_index[_start]
            stop_point=_index[_start+1]
            return T.sum(_race[start_point:stop_point])

        _cumsum, _ = theano.scan(cumsum_within_group,
                                    sequences=[T.arange(index.shape[0]-1)],
                                    non_sequences=[index, _raw_w])


        #构造一个rep(cumsum,times)的序列，目的是直接相除从而得到每匹马的概率
        _times, _ = theano.scan(fn=lambda i, index: index[i+1]-index[i],
                                sequences=[T.arange(index.shape[0]-1)],
                                non_sequences=index)

        _race_prob_div = T.repeat(_cumsum.ravel(), _times)

        self.race_prob = _raw_w / T.reshape(_race_prob_div,[_race_prob_div.shape[0],1])

        self.mean_neg_loglikelihood = None

        self.neg_log_likelihood = None

        self.r_square = None

    def negative_log_likelihood(self, index):

        _1st_prob, _ = theano.scan(fn= lambda _1st, prior_reuslt, _prob: prior_reuslt+T.log(_prob[_1st]),
                                   sequences=[index[:-1]],
                                   outputs_info=T.as_tensor_variable(numpy.array([0.])),
                                   non_sequences=self.race_prob)

        self.neg_log_likelihood = -_1st_prob[-1] #这个是负的

        self.mean_neg_loglikelihood = self.neg_log_likelihood/index.shape[0]

        return self.mean_neg_loglikelihood

    def Rsquare(self, index): #rsqaure约大越好，函数返回的值越小越好

        self.r_square = 1+self.neg_log_likelihood/(T.log(1./index[-1]))

        return -self.neg_log_likelihood/(T.log(1./index[-1]))


def cg_optimization_mnist(n_epochs=50, batch_size=100 ,dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'], n_in=27):

    #############
    # LOAD DATA #
    #############
    datasets = load_data(dataset[0],dataset[1],dataset[2])

    train_set_x, train_set_y, train_set_index = datasets[0]
    valid_set_x, valid_set_y, valid_set_index = datasets[1]
    test_set_x, test_set_y, test_set_index = datasets[2]

    batch_size = batch_size    # size of the minibatch

    n_train_batches = len(numpy.unique(train_set_index.eval())) / batch_size
    n_valid_batches = len(numpy.unique(valid_set_index.eval())) / batch_size
    n_test_batches = len(numpy.unique(test_set_index.eval())) / batch_size

    n_in = n_in  # number of features in a horse


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'


    minibatch_offset = T.lscalar()  #i * batch_size for i in xrange(n_train_batches)
    x = T.matrix()
    y = T.ivector()
    index = T.ivector()

    #construct symbolic model
    classifier = ConditionalLogisticRegression(input=x, n_in=27, index=index)

    cost = classifier.negative_log_likelihood(index)

    #根据一个Minibatch号码在test数据上计算error
    test_model = theano.function(
        [minibatch_offset],
        classifier.errors(y), #计算test set上的错误率
        givens={
            x: test_set_x[minibatch_offset:minibatch_offset + batch_size],
            y: test_set_y[minibatch_offset:minibatch_offset + batch_size]
        },
        name="test"
    )
    #根据一个Minibatch号码在valida数据上计算error
    validate_model = theano.function(
        [minibatch_offset],
        classifier.errors(y), #计算validate set上的错误率
        givens={
            x: valid_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: valid_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="validate"
    )

    # 封装一个根据特定minibatch号码计算likelihood cost的函数
    batch_cost = theano.function(
        [minibatch_offset],
        cost, #计算一个起始点开始接下来batch_size个函数的log-liklihood
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_cost"
    )

    # 封装一个根据特定minibatch号码计算gradient的函数
    batch_grad = theano.function(
        [minibatch_offset],
        T.grad(cost, classifier.theta), #让所有预测对的那个概率加起来尽量的高
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        name="batch_grad"
    )

    # 计算train数据上的cost函数
    def train_fn(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        train_losses = [batch_cost(i * batch_size)
                        for i in xrange(n_train_batches)] #在所有的batch上计算cost，然后输出均值
        return numpy.mean(train_losses)

    # 用预测对的所有likelihood为cost，计算gradient
    def train_fn_grad(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        grad = batch_grad(0)
        for i in xrange(1, n_train_batches): #在batch上累加gradient然后除以batch的个数
            grad += batch_grad(i * batch_size)
        return grad / n_train_batches

    validation_scores = [numpy.inf, 0] #用来记录最小的在validation和test上的loss

    # 计算在validation数据上的错误率，如果创了记录，那么就在test数据上测试
    def callback(theta_value):
        classifier.theta.set_value(theta_value, borrow=True)
        #compute the validation error
        validation_losses = [validate_model(i * batch_size) #计算每个batch上的错误率
                             for i in xrange(n_valid_batches)]
        this_validation_loss = numpy.mean(validation_losses)
        print('validation error %f %%' % (this_validation_loss * 100.,))

        # check if it is better then best validation score got until now
        if this_validation_loss < validation_scores[0]:
            # if so, replace the old one, and compute the score on the
            # testing dataset
            validation_scores[0] = this_validation_loss
            test_losses = [test_model(i * batch_size) #如果效果好就在test set上计算错误率
                           for i in xrange(n_test_batches)]
            validation_scores[1] = numpy.mean(test_losses)

    ###############
    # TRAIN MODEL #
    ###############

    # using scipy conjugate gradient optimizer
    import scipy.optimize
    print ("Optimizing using scipy.optimize.fmin_cg...")
    start_time = time.clock()
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((n_in + 1) * n_out, dtype=x.dtype),
        fprime=train_fn_grad,
        callback=callback,
        disp=0,
        maxiter=n_epochs
    )
    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, with '
            'test performance %f %%'
        )
        % (validation_scores[0] * 100., validation_scores[1] * 100.)
    )

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))


if __name__ == '__main__':
    cg_optimization_mnist(dataset=['horse_train.csv','horse_valid.csv','horse_test.csv'],n_in=27)




