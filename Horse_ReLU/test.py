#-*- encoding:utf-8 -*-
__author__ = 'Yang'

        from theano import *
        import theano.tensor as T

        index=T.ivector()
        _raw_w=T.matrix()

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
