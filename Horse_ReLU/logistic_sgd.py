#-*- encoding:utf-8 -*-
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object): #As for the logistic regression we will
                                  # first define the log-likelihood and then
                                  # the loss function as being the negative log-likelihood.

    def __init__(self, input, n_in, n_out, W=None,b=None):



        if W is None:
            self.W = theano.shared(
                value=numpy.zeros(
                    (n_in, n_out), #其实w的每一列就是一个分类器，softmax就是归一化一个输入在每个类的分类器上的得分
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(
                value=numpy.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            self.b=b


        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]) #把0,1,2,3..100中应该是的那个label的概率取出来

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def show_labels(self,extendinput):

        self.p_newy_given_x = T.nnet.softmax(T.dot(extendinput, self.W) + self.b)

        self.p_newy = T.cast(T.argmax(self.p_newy_given_x, axis=1),'int32')

        return (self.p_newy,T.max(self.p_newy_given_x, axis=1))

    def show_weights(self):

        return((self.W.get_value(),self.b.get_value()))



def load_data(trainset,validset,testset):

    #分别读入三个文件并share他们
    f=open(validset,"rb")
    data=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
    f.close()
    valid_set=(data[:,:-1],data[:,-1])

    f=open(testset,"rb")
    data=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
    f.close()
    test_set=(data[:,:-1],data[:,-1])

    f=open(trainset,"rb")
    data=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
    f.close()
    train_set=(data[:,:-1],data[:,-1])

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def sgd_optimization_mnist(learning_rate=0.01, n_epochs=1000,
                           dataset=[],batch_size=50,
                           newx=None,newy=None,
                           n_in=29,n_out=2,weights_file=None,bias_file=None):

    datasets = load_data(dataset[0],dataset[1],dataset[2])

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #shape[0] 5000
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'


    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    #logistic专用的cost函数
    cost = classifier.negative_log_likelihood(y)


    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 500000
    patience_increase = 2
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

    #############################
    #####extend the new features#
    #############################

    if newx is None:
        print "There is no extending data"
    else:
        print "Now extending"

        #read in data
        f=open(newx,"rb")
        newdata=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
        f.close()

        #share it to theano
        newinput = theano.shared(numpy.asarray(newdata,dtype=theano.config.floatX),borrow=True)


        #compile the function
        extend_model = theano.function(
        inputs=[],
        outputs=classifier.show_labels(newinput), #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
        )

        labels,prob=extend_model()
        print(labels)
        fmt = ",".join(["%i"] + ["%f"])
        numpy.savetxt(newy,zip(labels,prob),fmt=fmt,delimiter=',')


    ###########################
    #extact the weights########
    ###########################
    print "Extracting weights:"
    #label and weights should be extracted separately
    weights,bias=classifier.show_weights()
    #print(weights,bias)
    numpy.savetxt(weights_file,weights,delimiter=',')
    numpy.savetxt(bias_file,bias,delimiter=',')



def sgd_logistic_extend_as_you_want(newx=None,newy=None,W=None,b=None):

    print "Reading in the weights and bias"

    f=open(W,"rb")
    data=numpy.loadtxt(f,delimiter=',',dtype=float)
    f.close()
    shared_W = theano.shared(numpy.asarray(data,dtype=theano.config.floatX),borrow=True)

    f=open(b,"rb")
    data2=numpy.loadtxt(f,delimiter=',',dtype=float)
    f.close()
    shared_b = theano.shared(numpy.asarray(data2,dtype=theano.config.floatX),borrow=True)

    n_in=data.shape[0]
    n_out=data.shape[1]

    print "Reading in the new input"

    f=open(newx,"rb")
    newdata=numpy.loadtxt(f,delimiter=',',dtype=float,skiprows=1)
    f.close()
    newinput = theano.shared(numpy.asarray(newdata,dtype=theano.config.floatX),borrow=True)

    #build the new model
    x = T.matrix('x')
    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out, W=shared_W,b=shared_b)

    print "Extending on the new data"
    extend_model = theano.function(
    inputs=[],
    outputs=classifier.show_labels(newinput), #如果想要返回一个用theano计算的函数，那么一定要额外在主函数里建一个专门返回子函数的函数
    )

    labels,prob=extend_model()
    print(labels)
    fmt = ",".join(["%i"] + ["%f"])
    numpy.savetxt(newy,zip(labels,prob),fmt=fmt,delimiter=',')



if __name__ == '__main__':
    ##remeber: the label should start from 0
    sgd_optimization_mnist(learning_rate=0.2, n_epochs=200,n_in=11,n_out=7,
                           dataset=["/Users/Yang/Desktop/train_data.csv","/Users/Yang/Desktop/valid_data.csv",
                                    "/Users/Yang/Desktop/valid_data.csv"],
                           batch_size=2,
                           newx='/Users/Yang/Desktop/extend_data.csv',newy='/Users/Yang/Desktop/margin_result.csv',
                           weights_file='/Users/Yang/Desktop/weights.csv',bias_file='/Users/Yang/Desktop/bias.csv')

    sgd_logistic_extend_as_you_want(newx='/Users/Yang/Desktop/extend_data.csv',newy='/Users/Yang/Desktop/margin_result2.csv',
                                    W='/Users/Yang/Desktop/weights.csv',b='/Users/Yang/Desktop/bias.csv')
