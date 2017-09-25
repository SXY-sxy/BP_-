#-*-coding:utf-8 -*-
import random
import numpy as np
import random
from math import *
from numpy import *
import matplotlib.pyplot as plt


import operator
#from bpNet import *

class BPNet(object):
    '''
    def __init__(self):
    def logistic(self, net):
    def dlogit(self, net):
    def errorfunc(self, inX):
    def normallize(self, dataMat):
    def loadDataSet(self, filename):
    def addcol(self,matrix1, matrix2):
    def init_hiddenWB(self):
    def bpTrain(self):
    def BPClassfier(self, start, end, steps = 30):
    def classfyLine(self, plt, x, z):
    def TrendLine(self, plt, color = 'r'):
    def drawClassScatter(self, plt):
    '''

    def __init__(self):
        self.eb = 0.01
        self.iterator = 0
        self.eta = 0.1
        self.mc = 0.3
        self.maxiter = 2000
        self.nHidden = 4
        self.nOut = 1

        self.errlist = []
        self.dataMat = 0
        self.classLabels = 0
        self.nSampNum = 0
        self.nSampDim = 0

    #激活函数传递
    def logistic(self, net):
       return 1.0 / (1.0 + exp(-net))

    #全局误差函数
    def errorfunc(self, inX):
        return sum(power(inX, 2)*0.5)

    #传递函数导函数
    def dlogit(self, net):
        return multiply(net, (1.0 - net))

    #隐含层初始化
    def init_hiddenWB(self):
        self.hi_w = 2.0 * (random.rand(self.nHidden, self.nSampDim) - 0.5)
        self.hi_b = 2.0 * (random.rand(self.nHidden, 1) - 0.5)
        self.hi_wb = np.mat(self.addcol(np.mat(self.hi_w), np.mat(self.hi_b)))

    #输出层初始化
    def init_OutputWB(self):
        self.out_w = 2.0 * (random.rand(self.nOut, self.nHidden) - 0.5)
        self.out_b = 2.0 * (random.rand(self.nOut, 1) - 0.5)
        self.out_wb = 2.0 * (self.addcol(np.mat(self.out_w), np.mat(self.out_b)))


    #加载数据集
    def loadDataSet(self, filename):
        self.dataMat = []
        self.classLabels = []
        fr = open(filename)
        for line in fr.readlines():
            lineArr = line.strip().split()
            self.dataMat.append([float(lineArr[0]), float(lineArr[1]), 1.0])
            self.classLabels.append(int(lineArr[2]))

        self.dataMat = np.mat(self.dataMat)
        m,n = np.shape(self.dataMat)
        self.nSampNum = m
        self.nSampDim = n - 1
        return self.dataMat, self.classLabels

    #数据集归一化
    def normalize(self, dataMat):
        m, n = np.shape(dataMat)
        for i in xrange(n-1):
            dataMat[:,i] = (dataMat[:,i] - np.mean(dataMat[:,i])) / (np.std(dataMat[:,i]) + 1.0*e - 10)
            return dataMat


    #矩阵增加新列
    def addcol(self, matrix1, matrix2):
        m1, n1 = np.shape(matrix1)
        m2, n2 = np.shape(matrix2)
        if m1 != m2:
            print "different rows, can not merge matrix"
            return
        mergMat = np.zeros((m1, n1 + n2))
        mergMat[:, 0:n1] = matrix1[:, 0:n1]
        mergMat[:, n1:(n1+n2)] = matrix2[:,0:n2]
        return mergMat

    # 绘制分类点
    def drawClassScatter(self, plt):
        i = 0
        for mydata in self.dataMat:
            if self.classLabels[i] == 0:
                 plt.scatter(mydata[0, 0], mydata[0, 1], c='blue', marker='o')
            else:
                 plt.scatter(mydata[0, 0], mydata[0, 1], c='red', marker='s')
            i += 1


    #BP网络主程序
    def bpTrain(self):
        SampIn = self.dataMat.T
        expected = np.mat(self.classLabels)
        self.init_hiddenWB()
        self.init_OutputWB()
        dout_wbOld = 0.0
        dhi_wbOld = 0.0

        #主循环
        # 工作信息正向传播
        #信息从输入层到隐含层：这里使用了矢量计算， 计算的是整个样本集的结果。结果是4行307列的矩阵
        for i in xrange(self.maxiter):
            hi_input = self.hi_wb * SampIn
            hi_output = self.logistic(hi_input)
            hi2out = self.addcol(hi_output.T, np.ones((self.nSampNum, 1))).T

            out_input = np.dot(self.out_wb, hi2out)
            out_output = self.logistic(out_input)

            err = expected - out_output
            sse = self.errorfunc(err)
            self.errlist.append(sse)
            if sse <= self.eb:
                self.iterator = i + 1
                break

            DELTA = np.multiply(err, self.dlogit(out_output))
            delta = np.multiply(self.out_wb[:,:-1].T*DELTA, self.dlogit(hi_output))
            dout_wb = DELTA * hi2out.T
            dhi_wb = delta * SampIn.T


            if i == 0:
                self.out_wb = self.out_wb + self.eta * dout_wb
                self.hi_wb = self.hi_wb + self.eta * dhi_wb
            else:
                self.out_wb = self.out_wb + (1.0 - self.mc) *self.eta * dout_wb + self.mc * dout_wbOld
                self.hi_wb = self.hi_wb + (1.0 - self.mc) * self.eta*dhi_wb + self.mc * dhi_wbOld

            dout_wbOld = dout_wb
            dhi_wbOld = dhi_wb




    def BPClassfier(self, start, end, steps = 30):
        x = np.linspace(start, end, steps)
        xx = np.mat(np.ones((steps, steps)))
        xx[:,0:steps] = x
        yy = xx.T
        z = np.ones((len(xx), len(yy)))
        for i in range(len(xx)):
            for j in range(len(yy)):
                xi = []
                tauex =[]
                tautemp = []
                np.mat(xi.append([xx[i, j], yy[i, j], 1]))
                hi_input = self.hi_wb*(np.mat(xi).T)
                hi_out = self.logistic(hi_input)
                taumrow, taucol = np.shape(hi_out)
                tauex = np.mat(np.ones((1, taumrow+1)))
                tauex[:, 0:taumrow] = (hi_out.T)[:,0:taumrow]
                out_input = self.out_wb * (np.mat(tauex).T)
                out = self.logistic(out_input)
                z[i,j] = out
        return x, z


    def classfyLine(self,plt, x, z):
        plt.contour(x, x, z, 1, colors='black')

    def TrendLine(self,plt,color='r'):
        X = np.linspace(0, self.maxiter, self.maxiter)
        Y = np.log2(self.errlist)
        plt.plot(X, Y, color)
