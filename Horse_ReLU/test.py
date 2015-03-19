#-*- encoding:utf-8 -*-
__author__ = 'Yang'

        from theano import *
        import theano.tensor as T

        from logistic_cg import load_data



        batch_size=100
        dataset=['horse_valid.csv','horse_valid.csv','horse_valid.csv']
        datasets = load_data(dataset[0],dataset[1],dataset[2])

        train_set_x, train_set_y, train_set_index = datasets[0]
        valid_set_x, valid_set_y, valid_set_index = datasets[1]
        test_set_x, test_set_y, test_set_index = datasets[2]

        n_train_batches = (len(numpy.unique(train_set_index.eval()))-1) / batch_size #-1是因为多一个零
        n_valid_batches = (len(numpy.unique(valid_set_index.eval()))-1) / batch_size
        n_test_batches = (len(numpy.unique(test_set_index.eval()))-1) / batch_size

        n_in = train_set_x.shape[1].eval()  # number of features in a horse
        n_out = 1
        print '... building the model'

        minibatch = T.lscalar()
        x = T.matrix()
        index = T.ivector()
        classifier = ConditionalLogisticRegression(input=valid_set_x, n_in=n_in, index=valid_set_index)
        cost = classifier.negative_log_likelihood(valid_set_index)

        batch_grad = theano.function(
        [minibatch],
        T.grad(cost, classifier.theta), #这句有问题
        givens={
            x: train_set_x[train_set_index[minibatch]:train_set_index[minibatch + batch_size]],
            index: train_set_index[minibatch:(minibatch + batch_size + 1)] - train_set_index[minibatch]
        },
        name="batch_grad"
        ,mode="DebugMode"
        )



    from theano import *
    import theano.tensor as T
    from logistic_cg_fake import LogisticRegression
    from logistic_sgd_fake import load_data
    n_in=27
    n_out=1
    dataset=["special_horse.csv","special_horse.csv","special_horse.csv"]
    batch_size=2
    datasets = load_data(dataset[0],dataset[1],dataset[2])

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    batch_size = batch_size   # size of the minibatch

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    # allocate symbolic variables for the data
    minibatch_offset = T.lscalar()  # offset to the start of a [mini]batch
    x = T.matrix()   # the data is presented as rasterized images
    y = T.ivector()  # the labels are presented as 1D vector of
                     # [int] labels

    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    cost = classifier.negative_log_likelihood(y).mean()

    batch_grad = theano.function(
        [minibatch_offset],
        T.grad(cost, classifier.theta),
        givens={
            x: train_set_x[minibatch_offset: minibatch_offset + batch_size],
            y: train_set_y[minibatch_offset: minibatch_offset + batch_size]
        },
        mode="DebugMode",
        name="batch_grad"
    )















        from theano import *
        import theano.tensor as T
        import numpy

        index=T.ivector()
        _raw_w=T.matrix()

        #计算每组比赛内的和
        def cumsum_within_group(_start, _index, _race):
            start_point=_index[_start]
            stop_point=_index[_start+1]
            return T.sum(_race[start_point:stop_point], dtype='float32')

        _cumsum, _ = theano.scan(cumsum_within_group,
                                 sequences=[T.arange(index.shape[0]-1)],
                                 non_sequences=[index, _raw_w])


        #构造一个rep(cumsum,times)的序列，目的是直接相除从而得到每匹马的概率
        _times, _ = theano.scan(fn=lambda i, index: index[i+1]-index[i],
                                sequences=[T.arange(index.shape[0]-1)],
                                non_sequences=index)
        #
        # _output_info = T.alloc(_cumsum.ravel()[0],_times[0])
        # #_output_info2 = T.concatenate((_output_info, T.alloc(_cumsum.ravel()[1], _times[1])))
        #
        # _race_prob_div, _ = theano.scan(fn=lambda t, _pre, _all_prob, time: T.concatenate((_pre, T.alloc(_all_prob[t], time[t]))),
        #                                 sequences = [T.arange(_times.shape[0])[1:]],
        #                                 outputs_info = _output_info,
        #                                 non_sequences = [_cumsum.ravel(), _times])
        #
        # _race_prob_div = _race_prob_div[-1]

        _race_prob_div = T.repeat(_cumsum.ravel(),_times)

        race_prob = _raw_w / T.reshape(_race_prob_div,[_race_prob_div.shape[0],1])

        f=theano.function([_raw_w,index],_race_prob_div)

        #f2=theano.function([_raw_w,index],[_cumsum,_times])

        f3=theano.function([_raw_w, index],T.arange(_times.shape[0])[1:],on_unused_input='ignore' )
        x=numpy.array([1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8],'float32').reshape(8,1)

        y=numpy.array([0,2,5,8],'int32')


        result, _ = theano.scan(fn = lambda i: i ,
                                sequences = [index[:-1]])














        def cumsum_within_group(_start, _index, _race): #start: arange(len(index)-1)

            start_point=_index[_start]
            stop_point=_index[_start+1]

            return T.sum(_race[start_point:stop_point])


        _cumsum, _ = theano.scan(cumsum_within_group,
                                    sequences=[T.arange(index.shape[0]-1)],
                                    non_sequences=[index, _raw_w])



        _times, _=theano.scan(fn=lambda i, index: index[i+1]-index[i],
                              sequences=[T.arange(index.shape[0]-1)],
                              non_sequences=index)


        race_prob_div=T.repeat(_cumsum.ravel(),_times)
        race_prob=_raw_w/T.reshape(race_prob_div,[race_prob_div.shape[0],1])

        #f2=theano.function([_raw_w,index,_cumsum], _cumsum,on_unused_input='warn')
        #f3=theano.function([_raw_w,index,_cumsum], _times,on_unused_input='warn')
        #f4=theano.function([_raw_w,index,_cumsum], race_prob_div,on_unused_input='warn')
        f5=theano.function([_raw_w,index,], race_prob)


        def divide_within_all(_start,_previous, _cumsumm, _index, _race):

            start_point = _index[_start]
            stop_point = _index[_start+1]

            #因为不同长度的不让拼接，所以先放弃
            race_prob =_race[start_point:stop_point]/_cumsumm[_start]
            return T.concatenate([_previous.ravel(),race_prob.ravel()])


        _prob, _ = theano.scan(fn=divide_within_all,
                               sequences=[T.arange(_cumsum.shape[0])],
                               outputs_info=T.as_tensor_variable(numpy.array([0.])),
                               non_sequences=[_cumsum , index, _raw_w])

        prob=T.reshape(_prob,(_prob.ravel().shape[0],1))
        f1=function([index,_raw_w],_cumsum)
        f2=function([_cumsum,index,_raw_w],_prob)


        import numpy as np
        from theano import *
        import theano.tensor as T
        from logistic_cg import load_data
        input=T.matrix()
        n_in=27
        n_out=1 #对于CL模型来说，并不是每一类构建一个分类平面，一直都只有一个数值

        theta = theano.shared(
            value=numpy.zeros(
                n_in * n_out + n_out,
                dtype=theano.config.floatX
            ),
            name='theta',
            borrow=True
        )
        W = theta[0:n_in * n_out].reshape((n_in, n_out))
        b = theta[n_in * n_out:n_in * n_out + n_out]

        # 计算概率的exp归一化矩阵,group by index
        _raw_w = T.exp(T.dot(input, W) + b)
        f=function([input],_raw_w)

        dataset=['horse_train.csv','horse_valid.csv','horse_test.csv']
        datasets = load_data(dataset[0],dataset[1],dataset[2])
        test_set_x, test_set_y, test_set_index = datasets[2]
        f=function([],test_set_x)

        index=T.ivector()
        race_prob=T.matrix()
        _1st_prob, _ = theano.scan(fn= lambda _1st, prior_reuslt, _prob: prior_reuslt+T.log(_prob[_1st]),
                                   sequences=[index[:-1]],
                                   outputs_info=T.as_tensor_variable(numpy.array([0.])),
                                   non_sequences=race_prob)
        f=theano.function([race_prob,index],-_1st_prob[-1]/T.log(1./index[-1]) )


        init_y = T.alloc(numpy.cast[theano.config.floatX](0), [2,2,2])


        _times=T.ivector()
        _cumsum=T.matrix()

        _output_info = T.alloc(_cumsum.ravel()[0],_times[0])
        _race_prob_div, _ = theano.scan(fn=lambda t, _pre, prob, time: T.concatenate((_pre, T.alloc(prob[t], time[t]))),
                                        sequences = [T.arange(_times.shape[0])[1:]],
                                        outputs_info = _output_info,
                                        non_sequences = [_cumsum.ravel(), _times])

        _race_prob_div = _race_prob_div[-1]

        race_prob = T.reshape(T.repeat(1.2,7),[7,1]) / T.reshape(_race_prob_div,[_race_prob_div.shape[0],1])

        f=theano.function([_cumsum, _times],race_prob)


        d=T.dvector()
        f=theano.function([d],T.mean(d.ravel(),dtype='float32'))




from theano import *
import theano.tensor as T
from logistic_cg import load_data

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
        self._raw_w = T.exp(T.dot(input, self.W) + self.b)

        #计算每组比赛内的和
        def cumsum_within_group(_start, _index, _race):
            start_point = _index[_start]
            stop_point = _index[_start+1]
            return T.sum(_race[start_point:stop_point], dtype='float32')

        self._cumsum, _ = theano.scan(cumsum_within_group,
                                 sequences=[T.arange(index.shape[0]-1)],
                                 non_sequences=[index, self._raw_w])


        #构造一个rep(cumsum,times)的序列，目的是直接相除从而得到每匹马的概率
        self._times, _ = theano.scan(fn=lambda i, index: index[i+1]-index[i],
                                sequences=[T.arange(index.shape[0]-1)],
                                non_sequences=index)

        # _output_info = T.alloc(_cumsum.ravel()[0],_times[0])
        # _race_prob_div, _ = theano.scan(fn=lambda t, _pre, prob, time: T.concatenate((_pre, T.alloc(prob[t], time[t]))),
        #                                 sequences = [T.arange(_times.shape[0])[1:]],
        #                                 outputs_info = _output_info,
        #                                 non_sequences = [_cumsum.ravel(), _times])
        # _race_prob_div = _race_prob_div[-1]

        # self._race_prob_div = T.repeat(self._cumsum.ravel(), self._times)
        #
        # self.race_prob = self._raw_w / T.reshape(self._race_prob_div,[self._race_prob_div.shape[0], 1])

        #self.race_prob = _raw_w / T.min(_cumsum)

        self._race_prob_div = T.ones_like(self._raw_w)

        def change_race_prob_div(_i, _change, _rep, _times, _item):
            _change = T.set_subtensor(_change[_rep[_i]:_rep[_i+1]], T.reshape(T.alloc(_item[_i],_times[_i]),(_times[_i],1)))
            return _change

        self._k, __ = theano.scan(fn = change_race_prob_div,
                            sequences=[T.arange(index.shape[0]-1)],
                            outputs_info=[self._race_prob_div],
                            non_sequences=[index,self._times, self._cumsum])

        self.race_prob = self._raw_w / self._k[-1]

        self.mean_neg_loglikelihood = None

        self.neg_log_likelihood = None

        self.pos_log_likelihood=None

        self.r_square = None

        self.r_error = None

    def negative_log_likelihood(self, index):

        _output_info = T.as_tensor_variable(numpy.array([0.]))

        _output_info = T.unbroadcast(_output_info, 0)

        _1st_prob, _ = theano.scan(fn= lambda _1st, prior_reuslt, _prob: prior_reuslt+T.log(_prob[_1st]),
                                   sequences=[index[:-1]],
                                   outputs_info=_output_info, #特别注意：output_info一定不能用numpy组成的序列，用shared或者禁掉broadcast
                                   non_sequences=self.race_prob)

        self.neg_log_likelihood = 0. - _1st_prob[-1] #这个是负的

        self.pos_log_likelihood = _1st_prob[-1]

        self.mean_neg_loglikelihood = self.neg_log_likelihood/(index.shape[0]-1)

        return T.mean(self.mean_neg_loglikelihood.ravel(), dtype='float32')

        #return T.mean(self.neg_log_likelihood.ravel(), dtype='float32')

        # _1st_prob, _ = theano.scan(fn= lambda _1st,  _prob: T.log(_prob[_1st]),
        #                            sequences=[index[:-1]],
        #                            non_sequences=self.race_prob)
        #
        # self.neg_log_likelihood = -T.sum(_1st_prob.ravel(), dtype='float32')
        #
        # self.mean_neg_loglikelihood = -T.mean(_1st_prob.ravel(), dtype='float32')

    def Rsquare(self, index): #rsqaure约大越好，函数返回的值越小越好

        _output_info = T.as_tensor_variable(numpy.array([0.]))

        _output_info = T.unbroadcast(_output_info, 0)

        _r_square_div, _ = theano.scan(fn = lambda _t, prior_reuslt: prior_reuslt+T.log(1./_t),
                                       sequences=[self._times],
                                       outputs_info=_output_info #特别注意：output_info一定不能用numpy组成的序列，用shared或者禁掉broadcast
                                       )

        self.r_error = self.pos_log_likelihood / _r_square_div[-1]

        self.r_square = 1 - self.r_error

        return T.mean(self.r_error.ravel(), dtype='float32')

batch_size=100
dataset=['horse_valid.csv','horse_valid.csv','horse_valid.csv']
datasets = load_data(dataset[0],dataset[1],dataset[2])

train_set_x, train_set_y, train_set_index = datasets[0]
valid_set_x, valid_set_y, valid_set_index = datasets[1]
test_set_x, test_set_y, test_set_index = datasets[2]

n_train_batches = (len(numpy.unique(train_set_index.eval()))-1) / batch_size #-1是因为多一个零
n_valid_batches = (len(numpy.unique(valid_set_index.eval()))-1) / batch_size
n_test_batches = (len(numpy.unique(test_set_index.eval()))-1) / batch_size

n_in = train_set_x.shape[1].eval()  # number of features in a horse
n_out = 1
print '... building the model'

minibatch = T.lscalar()
x = T.matrix()
index = T.ivector()
classifier = ConditionalLogisticRegression(input=valid_set_x, n_in=n_in, index=valid_set_index)
cost = classifier.negative_log_likelihood(valid_set_index)
error = classifier.Rsquare(valid_set_index)

classifier.theta.eval()
classifier.W.eval()
classifier.b.eval()

classifier._raw_w.eval()