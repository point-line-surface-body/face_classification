import numpy
import pylab
import scipy
from scipy.misc import imsave
import os
import glob


def metrica_evklida(x,y):
    return numpy.sqrt(sum((x-y)**2))


def PCA(x):
    m = numpy.average(x,axis=0)
    m = numpy.average(m,axis=0)         #mean of vectors
    x0 = x - m                          #centered vectors
    r = numpy.matrix(numpy.zeros(shape = (img.shape[0]**2,img.shape[0]**2)))
    R = numpy.matrix(numpy.zeros(shape = (img.shape[0]**2,img.shape[0]**2)))

    x01 = x0
    for i in xrange(num_man):
        x01[i] = numpy.matrix(x0[i])
        r = numpy.dot(x01[i].T,x01[i])      #interclass autocorrelation matrix
        r /= len(x0[i])                     #num faces in class
        R += r
    R /= num_man                #num class
                                #autocorrelation matrix all vectors

    lmbd,e = numpy.linalg.eig(R)    #eigenvalues and eigenvectors
    e = e.T
    sr = numpy.average(lmbd)
    S = []
    k = 0
    for i in range(len(lmbd)):          #Kaiser's rule
        if (lmbd[i] > sr):
            S.append(e[i])
            k += 1
    S = numpy.array(S).reshape(k,len(x0[0][0]))          #projection matrix
    print S

    x_new = numpy.zeros(shape = (num_man,len(x0[0]),k))
    for i in xrange(num_man):
        for j in xrange(len(x0[i])):
            x_new[i][j] = numpy.dot(S,x0[i][j])#result vector - proektsia centrirovannih
    x_new = numpy.array(x_new)            #vectorov na prostranstvo menshei razmernosti  
    return S,x_new


def nearest_neighbour(path):
    q = 10*numpy.ones(shape=(len(x)))
    l = -1
    img_in = pylab.imread(path)
    f_all = x1_new
    S = S1
    f_new = numpy.array(img_in.flatten())
    m1 = numpy.average(x,axis=0)
    m1 = numpy.average(m1,axis=0)
    f0_new = f_new - m1
    f0_new = numpy.dot(S,f0_new)
    for i in xrange(len(x)):
        for j in xrange(len(x[i])):
            q[i] = min(q[i],metrica_evklida(f_all[i][j],f0_new))
        l = numpy.argmin(q)
    img_out = numpy.zeros(shape = (img_in.shape[0], img_in.shape[1]))
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            img_out[i,j] = x[l][0][i*img_in.shape[0] + j]
    pylab.imshow(img_out,cmap = 'gray')
    pylab.show()



def drawEigVector(mas,g):
    img_eigvect = numpy.zeros(shape = (img.shape[0], img.shape[1]))
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            img_eigvect[i,j] = mas[i*img.shape[0] + j]
    scipy.misc.imsave(os.getcwd() + '\\eigvector\\' + str(g) + '.png',img_eigvect)


###################################MAIN########################################
listfaces = glob.glob('*.png')
print(len(listfaces))

x = []
i_human = 0
for i in xrange(len(listfaces)):
    if(i < 10):
        faces = glob.glob(os.getcwd() + '\subject' + '0'+ str(i) + '*')
    else:
        faces = glob.glob(os.getcwd() + '\subject' + str(i) + '*')
    if(not faces):
        continue
    x.append([])
    for path in faces:
        img = pylab.imread(path)
        imglent = numpy.array(img).reshape((img.shape[0]*img.shape[1]))
        x[i_human].append(imglent)
    i_human += 1
num_man = len(x)


S1 , x1_new = PCA(x)

imagepath = 'subject02.surprisedHB0_00S0ed40sz_100le30_30.bmp.png'
nearest_neighbour(imagepath)









