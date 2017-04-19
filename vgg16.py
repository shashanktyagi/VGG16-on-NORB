import tensorflow as tf
import os
import numpy as np
from scipy.misc import imread, imresize, imshow
from Models import vgg16
from pdb import set_trace as breakpoint

def get_train_data():
    fid_images = open('./NORB/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat','r')
    fid_labels = open('./NORB/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat','r')

    for i in xrange(6):
        a = fid_images.read(4)    #header

    num_images = 24300*2
    images = np.zeros((num_images,96,96))

    for idx in xrange(num_images):
        temp = fid_images.read(96*96)
        images[idx,:,:] = np.fromstring(temp,'uint8').reshape(96,96).T 

    for i in xrange(5):
        a = fid_labels.read(4) #header

    labels = np.fromstring(fid_labels.read(num_images*np.dtype('int32').itemsize),'int32')
    labels = np.repeat(labels,2)

    perm = np.random.permutation(num_images)
    images = images[perm]
    labels = labels[perm]
    labels = labels.reshape(images.shape[0],1) == np.arange(5) # one hot
    #imshow(images[2331,:,:])
    
    return images,labels    

def get_test_data():
    fid_images = open('./NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r')
    fid_labels = open('./NORB/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r')

    for i in xrange(6):
        a = fid_images.read(4)    #header

    num_images = 24300*2
    images = np.zeros((num_images,96,96))

    for idx in xrange(num_images):
        temp = fid_images.read(96*96)
        images[idx,:,:] = np.fromstring(temp,'uint8').reshape(96,96).T 

    for i in xrange(5):
        a = fid_labels.read(4) #header

    labels = np.fromstring(fid_labels.read(num_images*np.dtype('int32').itemsize),'int32')
    labels = np.repeat(labels,2)

    perm = np.random.permutation(num_images)
    images = images[perm]
    labels = labels[perm]
    labels = labels.reshape(images.shape[0],1) == np.arange(5) # one hot
    #imshow(images[2331,:,:])
    
    return images,labels 


def get_accuracy(sess, accuracy, X, Y):
    batch_size = 200
    num_batches = int(X.shape[0]/batch_size)
    acc = 0
    for i in xrange(num_batches):
        idxs = i*batch_size
        idxe = idxs + batch_size
        batch_x = X[idxs:idxe,:,:,:] 
        batch_y = Y[idxs:idxe,:]
        acc += sess.run(accuracy, feed_dict={imgs: batch_x, labels: batch_y,  dropout: 1})

    acc /= num_batches
    return acc


if __name__ == '__main__':

    curr_dir = os.path.dirname(os.path.realpath(__file__))

    X_train,Y_train = get_train_data()
    print 'Training data loaded'
    X_train = np.repeat(X_train[:,:,:,np.newaxis],3,axis=3)
    
    #TODO: subtract image mean

    train_size = 40000
    X = X_train[0:train_size,:,:,:]
    Y = Y_train[0:train_size,:]
    X_val = X_train[train_size:48600,:,:,:]
    Y_val = Y_train[train_size:48600,:]


    learning_rate = 1e-4
    num_epochs = 30
    batch_size = 100
    display_step = 1
    p_dropout = 0.5

    imgs = tf.placeholder(tf.float32, [None, 96, 96, 3])
    labels = tf.placeholder(tf.float32, [None, 5])
    dropout = tf.placeholder(tf.float32,[])

    
    
    with tf.Session() as sess:
        vgg = vgg16(imgs, 'vgg16_weights.npz', sess, dropout, skip_layers=['fc6_W','fc6_b','fc7_W','fc7_b','fc8_W','fc8_b'])      
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3, labels=labels))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        vgg.load_weights('vgg16_weights.npz', sess, skip_layers=['fc6_W','fc6_b','fc7_W','fc7_b','fc8_W','fc8_b'])
        saver = tf.train.Saver()
        
        for epoch in xrange(num_epochs):
            #breakpoint()
            perm = np.random.permutation(X.shape[0])
            X = X[perm,:,:,:]
            Y = Y[perm,:]
            
            num_batches = int(X.shape[0]/batch_size)
            curr_loss = 0
            for i in xrange(num_batches):
                idxs = i*batch_size
                idxe = idxs + batch_size
                batch_x = X[idxs:idxe,:,:,:] 
                batch_y = Y[idxs:idxe,:]
                _, l = sess.run([optimizer,loss], feed_dict={imgs: batch_x, labels: batch_y, dropout: p_dropout})
                curr_loss += l/batch_size
                #print 'Epoch: %02d Iter: %03d loss: %.8f' % (epoch+1, i+1, curr_loss)    
            if (epoch+1)%display_step == 0:
                print 'Epoch: %02d  loss: %.4f' % (epoch+1,curr_loss)    
        saver.save(sess, curr_dir + '/trained_vgg.chkp')

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(vgg.fc3,1),tf.argmax(labels,1)),tf.float32))
        breakpoint()

        print 'Training Accuracy: %.4f' % (get_accuracy(sess, accuracy, X_train,Y_train))
        print 'Validation Accuracy: %.4f' % (get_accuracy(sess, accuracy, X_val, Y_val))

        #print 'Training Accuracy: %.4f' % (sess.run(accuracy, feed_dict={imgs: X[0:5000,:,:,:], labels:Y[0:5000,:],  dropout: 1}))
        #print 'Validation Accuracy: %.4f' % (sess.run(accuracy, feed_dict={imgs: X_val, labels:Y_val,  dropout: 1}))
        
        X_test,Y_test = get_test_data()
        X_test = np.repeat(X_test[:,:,:,np.newaxis],3,axis=3)
        print 'Test data loaded'
        print 'Test Accuracy: %.4f' % (get_accuracy(sess, accuracy, X_test, Y_test))
        breakpoint()

    
    




    
