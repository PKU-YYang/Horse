#-*- encoding:utf-8 -*-
__author__ = 'Yang'

        from theano import *
        import theano.tensor as T
        from logistic_cg import ConditionalLogisticRegression
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
        classifier = ConditionalLogisticRegression(input=x, n_in=n_in, index=index)
        cost = classifier.negative_log_likelihood(index).mean()

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

        index=T.ivector()
        race_prob=T.matrix()
        _1st_prob, _ = theano.scan(fn= lambda _1st, prior_reuslt, _prob: prior_reuslt+T.log(_prob[_1st]),
                                   sequences=[index[:-1]],
                                   outputs_info=T.as_tensor_variable(numpy.array([0.])),
                                   non_sequences=race_prob)
        f=theano.function([race_prob,index],-_1st_prob[-1]/T.log(1./index[-1]) )


        init_y = T.alloc(numpy.cast[theano.config.floatX](0), [2,2,2])