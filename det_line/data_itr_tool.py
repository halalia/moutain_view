'''
实现一个类，对于山脊线标注数据、山脊线的canny检测数据进行patch的输出
已有的全部函数方式实现的重写、搬运、改造

另一种，混合图像局部组成一个“图像”，这个在像素级别的领域似乎是混合多样性的方法，避免batch限制
但是上下文就有影响了
'''
import cv2,os,fnmatch,random,math
import numpy as np
import tensorflow as tf
from multiprocessing import freeze_support, Pool
from multiprocessing.managers import BaseManager, BaseProxy
import  itertools as itl
'''
# tf.data使用生成器作为数据来源的例子

sequence = np.array([[[1]],[[2],[3]],[[3],[4],[5]]])

def generator():
    for el in sequence:
        yield el

dataset = tf.data.Dataset().batch(1).from_generator(generator,
                                                    output_types= tf.int64, 
                                                    output_shapes=(tf.TensorShape([None, 1])))

iter = dataset.make_initializable_iterator()
el = iter.get_next()

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))
----------------------------------
[[1]]
[[2]
 [3]]
[[3]
 [4]
 [5]]
'''
class Canny_Method(object):
    '''
    这个类含有canny相关参数设定
    内部是函数以及相关参数的封装
    返回一个函数对象？还是在这个内部处理？
    
    '''
    def __init__(self,**params):
        '''
        创建的时候需要显式指明，参数名称需要符合对应规范
        可以控制参数参见下面的解析过程
        需要注意的是关于图像边界扩展的参数border_arg
        '''
        #print(type(param))
        #print(param)
        self.canny_hyst_1   = params['canny_hyst_1'] if 'canny_hyst_1' in params \
                            else 3
        self.canny_hyst_diff  = params['canny_hyst_diff'] if 'canny_hyst_diff' in params\
                                else 3
        self.canny_hyst_2 = self.canny_hyst_1 + self.canny_hyst_diff
        self.canny_edge  = params['canny_edge'] if 'canny_edge' in params \
                           else None
        self.canny_aperture  = params['canny_aperture']  if 'canny_aperture' in params\
                               else 3
        self.cannY_l2grad  = params['cannY_l2grad'] if 'cannY_l2grad' in params\
                             else 0
        self.border_arg  = params['border_arg'] if 'border_arg' in params\
                           else 1
        self.border_type  = params['border_type'] if 'border_type' in params\
                            else cv2.BORDER_REPLICATE
    def _canny_run(self,imput):
        # 输入若是3通道则转换为灰度
        # 单通道直接处理
        # partial 不能实现这种复杂的
        if imput.shape[2] ==3:
            # 注意，np split产生的shape中最后是（h,w,1）而不是(h，w)
            imput = cv2.cvtColor(imput,cv2.COLOR_BGR2GRAY)

        return cv2.Canny( cv2.copyMakeBorder(imput,
                                            self.border_arg,
                                            self.border_arg,
                                            self.border_arg,
                                            self.border_arg,
                                            self.border_type),
                           self.canny_hyst_1,
                           self.canny_hyst_2,
                           self.canny_edge,
                           self.canny_aperture,
                           self.cannY_l2grad)[ self.border_arg:-self.border_arg,
                                               self.border_arg:-self.border_arg
                                             ]
    
    def run(self,img):
        # 使用的时候，直接 a = Canny_Method(); handle = a.run; data = handle(img)
        # 作为类方法，直接 h = Canny_Method.run; data = h(img)
        #             注意：h = Canny_Method(canny_hyst_1 = 123).run; data = h(img)
        #                               但是这样就没有能够控制相关类参数成员相关的行为了
        # 累方法到底能不能使用类属性？？？，那么是怎么实现对于参数的控制的？？？
        fine_canny = self._canny_run(img)
        return fine_canny
    
    def run_on_channels(self,img):
        # 使用各种通道的边缘,输入拆分通道，输出与之对应
        B_canny = self._canny_run(np.dsplit(img,3)[0])
        G_canny = self._canny_run(np.dsplit(img,3)[1])
        R_canny = self._canny_run(np.dsplit(img,3)[2])
        # 合成
        return R_canny,G_canny,B_canny




class ridge_data(object):
    '''
    使用不同方法，实现新旧功能
    核心是个文件名称对子的生成器
    围绕这个生成器，可以获取单个图像、可以获取图像的batch
    
    （获取图像的局部patch的功能废弃了，
    原因是局部循环太慢，但是思路是正确的
    进行的判断本质上仅仅是利用图像局部范围的性质判断，
    还不认为是一种全局的语义性质的特性
    但是也不能完全否认全局性质的可能）
    
    对于新实现：-------------
    不进行图像局部patch的拆装，那样太慢。而是实现图像——————标注对子的给出。
    datas = ridge_data( test = False, 
                        data_dir = 'D:\_work\python\yuelushan\imagez\large\labled')
    data,anno= datas.get_data()
    训练模式下，给出原始图像、对应标记图
    测试模式下，给出原始图像、None
    
    对于旧实现，-------------
    并非没有用处，能够实现数据的增广，
    但是patch需要更大，裁切方式也不同、遍历方式也不同
    
    分类的是图像局部patch
    datas = ridge_data( test = False, 
                        data_dir = 'D:\_work\python\yuelushan\imagez\large\labled')
    data_provider = datas.get_patch_gen_XXX
    patch, label, pos = next(data_provider)
    约定，原始数据格式是png
    
    '''
    
    #  border_arg相关的截取局部参数，
    #       得到的结果应该是奇数更理想。
    #       位于patch from image中，另外在batch组装中也修改两处初始化
    # 从原本的0:a:2a，变为1:a:2a，保证a是中心
    def __init__(self,**dargs):
        # 参数用法，
        # args对应dataDir
        # dargs一般是run_mode = "train"\"test"\'predict'
        
        print(dargs)
        self.train_mode = None
        if ('run_mode' in dargs):
            print(r' 模式得到参数:',dargs['run_mode'])
            if dargs['run_mode'] in ["train","test",'predict']:            
                self.train_mode = dargs['run_mode']  else None
        if self.train_mode == None:
            raise ValueError(r"需设定模式")
        print('running_mode is: ',self.train_mode)
        
        # 决定局部取出尺寸大小(老设计)
        border_arg = dargs['border_arg'] if 'border_arg' in dargs else 12 # pop产生keyerror
        self.patch_sz = 2*border_arg -1 # 局部的长宽
        # 用于拓展边缘、采样的时候在靠边位置实现采样
        
        self.augment = dargs['augment'] if 'augment' in dargs else False
        # 是不是对于图像进行增殖处理。默认不进行

        self.batch_sz = dargs['batch_sz'] if 'batch_sz' in dargs else 16
        # 全图方式，数据批次大小 单一图像数据大小不小，这个数字不能大。装不下
        
        self.norm_batch = dargs['norm_batch'] if 'norm_batch' in dargs else 1 
        # 当前批次的预处理，归一化模式
        # 忘记了 取值1、2、3、0
        # 0，无处理
        # 1，仅仅处理均值、方差
        # 2，增加白化
        
        self.data_dir = dargs['data_dir'] if 'data_dir' in dargs else r"D:\_work\data_sets\dem\imagez\labeled1st"
#        self.data_source_list = self._get_fn_tuple_list()
#        self.tuple_gen = self._get_tuple()
        
        
        self.crop_to = dargs['crop_to'] if 'crop_to' in dargs else (512,512)
        # crop_to = （512，512）
        # 将图像局部缩放得到较大尺寸的，
        # 第一作为数据增值的一种作法
        # 第二作为训练数据尺寸可能不一样的时候的一种“统一手段”
        # 作为一种分割模型，不需要尺寸一致。但是训练的时候需要是一致的！！！1
        # 另外分类中，使用全局pooling，也能避免大小问题
        self.ntuple_gen = self._get_name_tuple2()# 名称的产生器



    def _get_name_tuple2(self,trim=False):
        '''
        包含了另外两个函数的功能，作为简化
        文件名称解析
        一个训练数据对，名称放在一个tuple中
        所有的对子放在一个list中
        
        都不是含有路径的全名，是baseName
        '''
        file_tuple_list = []
        all_list = os.listdir(self.data_dir)
        
        if self.train_mode in ["train",'test']:
            print(r'请确认标注文件名使用——anno结尾')
            anno_list = fnmatch.filter(all_list,'*_anno*')            
            for a_anno in anno_list:
                fname = a_anno.split('_anno')[0]
                full_name = '.'.join((fname,'png'))
                if os.path.isfile( os.path.join(self.data_dir,full_name)  ):
                    the_tuple = (full_name,a_anno)
                    file_tuple_list.append(the_tuple)
        else:# predict模式，处理这个文件夹中的所有图像，进行山脊线预测
            for a_name in all_list:
                if os.path.splitext(a_name)[1] in ['.jpg','.jpeg','.png']:# 图像名称
                    the_tuple = (full_name,None)
                    file_tuple_list.append(the_tuple)
        random.shuffle(file_tuple_list)
        
        if trim:# 整理文件，没用
            import shutil
            trim_path = os.path.join(self.data_dir,'trim')
            for a_tuple in file_tuple_list:
                shutil.copy2( os.path.join(self.data_dir,
                                           a_tuple[0]),
                              os.path.join(trim_path ,
                                           a_tuple[0]))       
                shutil.copy2( os.path.join(self.data_dir,
                                           a_tuple[1]),
                              os.path.join(trim_path ,
                                           a_tuple[1]))                
                
        for a_tuple in file_tuple_list:
            yield a_tuple# 能不能缩减成为一句？？
        raise(GeneratorExit('数据名称遍历结束'))
        

    def _patch_frm_a_img(self,img, lable_img, **dic_arg):
        '''
        废弃 

        从一个图像中随机选择各个列，
        各个列中随机选择存在的边缘点
        给出点坐标、给出图像邻域局部
        img           原始图像
        lable_img     含有标注信息的图像 训练中是数据，测试中是空数据
        字典参数      主要用于表示border_arg，关系了局部的大小、边界拓展的大小
        生成器，不断的给出图像中的某个局部,另外还有标签信息、位置信息。
        对于图像进行细微的边缘处理，得到所有边缘，
        遍历所有边缘点，获取相关的局部图像
        按照对应的标签，获取同位置标签。
        关于标签信息：
            在训练中，参照标签图像给出图像局部矩阵、对应标签、对应局部的位置。
            在执行测试中，给出局部
        关于位置信息：
            训练中，没有作用
            执行测试时，这个位置将会根据判别标签被记录。
        
        处理图像中相似性较高问题，使用随机方法采样，而不是顺序得到（相邻点的相似性是最高的）      
        '''        
        if (type(lable_img) == type(None)):
            # 没有标签数据，这是执行模式
            # 那么遍历的“引导数据”就是canny产物
            lable_img = Canny_Method().run(img)
        if (len(lable_img.shape)!=2):
            raise  AssertionError(' lable map should be single tunnel')
            # 注意raise之后还会正常运行,所以使用return退出函数
            return None
        if img.shape[0:2] != lable_img.shape:
            raise  AssertionError('input size should equals lable masp size')
            return None
    
        if 'border_arg' in dic_arg:
            border_arg = dic_arg['border_arg']
        else:
            border_arg = 32# 中间点各方向推这么多。对应64X64小突
            
        canny_img = Canny_Method().run(img)    
        # 遍历边缘数据。
        #tranvers_trace = lable_img.copy()
        subs_list = np.where(canny_img>0)
        # subs = [zip(subs_list[0],subs_list[1])]
        total_points = subs_list[0].shape[0]
        rand_seq = np.random.permutation(total_points)
    ##    while np.sum(np.zeros_like(tranvers_trace)) != 0:
    ##        rand_num = np.random.
        #图像进行边缘扩展，防止在边缘取周边的时候，出现错误
        img = cv2.copyMakeBorder(img,
                                border_arg,
                                border_arg,
                                border_arg,
                                border_arg,
                                cv2.BORDER_REFLECT)
        for i in rand_seq:
            pos = [subs_list[0][i],subs_list[1][i]]
            lable = lable_img[pos[0],pos[1]]
    ##        patch = img[pos[0]+border_arg-32:pos[0]+border_arg+32,
    ##                    pos[1]+border_arg-32:pos[1]+border_arg+32]  
            # NOTE: 由于图像进行过扩展，所以坐标需要移动  
            # 提供的坐标是原始的，但是图像是扩展了的 修改方法，冒号前加一pos[0]:变成pos[0]+1:
            patch = img[pos[0]+1:pos[0]+border_arg+border_arg,
                        pos[1]+1:pos[1]+border_arg+border_arg]  # 由于图像进行过扩展，所以坐标需要移动
            yield patch ,lable, pos
        raise GeneratorExit('this image tranversed')
    def _rotate_bound(self, image, angle):
        # 保留中心、保留边缘自动调整大小的旋转 
        #
        # 角度制
        # 中心点旋转，否则左上角旋转，丢失内容
        # 测试通过
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
     
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
     
        # 算出新的大小
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
     
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
     
        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
        
    def _data_augment(self, inputdata):
        '''实现数据增殖
        适用于这个任务包括了
        1，适度缩放  这个免了，标注数据模糊
        2，适度旋转
        3，左右反折
        4，基本的局部裁切实现
        '''
        if inputdata == None:
            return None
            # 存在测试模式中含有None的情况，这时标签是None
        else:
            # 随机创建上述的变形参数
            # 1 将数据随机旋转，旋转参数，并且得到的图像是在某个扩大框内
            # 2，设定随机的位置、长宽参数，进行截取
            # 3，随机决定是不是进行左右反折
            # 4，后处理，标签模糊问题，一定的阈值。
            # 创建M矩阵，2X3，适用于warpAffine。内部的旋转、缩放参数可以控制（平移省略）
            # 实际上从训练特征的局部性看（无论是不是显示的拆分patch，都不是整体图像对应一个类别，而是局部对局部。
            # 都对于平移没有意义、因为显示解patch、隐式解patch都是局部范围内的）
            # 所以有意义的基本就是
            # 旋转、反折。简单的重复，对于这个任务没有负面影响，因为本质还是会像素的
            if (np.random.rand()-0.5)>0 :# 随机的左右反折
                inputdata = np.fliplr(inputdata)
            rot_angle_r = np.random.rand()-0.5# 0中心，正负0.5
            rot_max = 5# 单位度
            rot_angle_d = rot_angle_r*rot_max/0.5
            # print('using augment,rotation %d'%(rot_angle_d))
            return self._rotate_bound(inputdata, rot_angle_d)
            # 局部截取是不是可以不要？因为毕竟这里没有标注边缘
            
    def _get_img(self,name):
        '''实现了对于名称为空的兼容
        前面名称可能有None'''
        if Name == None:
            return None
        else:
            return cv2.imread(os.path.join(self.data_dir,name),-1)
    def _gen_crop(self,image):
        '''
        对数据进行裁切,生成方式yield
        '''
        
        if image == None:
            yield None
        else:
            datashape = image.shape
            h_times = (datashape[0]/self.crop_to[0])
            w_times = (datashape[1]/self.crop_to[1])        
            # 若是有维度小于设定，则图像扩大
            if min(h_times, w_times)<1:
                times = 1/min(h_times, w_times)
                image = cv2.resize(src, None, dst=None, 
                                   fx=times, fy=times, 
                                   interpolation=cv2.INTER_CUBIC)# 放大不能使用area
            # 处理两个维度都不小于的情况
            datashape = image.shape
            h_times = math.ceil(datashape[0]/self.crop_to[0])
            w_times = math.ceil(datashape[1]/self.crop_to[1]) 
            h_step = math.floor(datashape[0] - self.crop_to[0])/(h_times)
            w_step = math.floor(datashape[1] - self.crop_to[1])/(w_times)
            for i in range(h_times):
                for j in range(w_times):
                    region = np.s_[i*h_step : i*h_step +self.crop_to[0] , 
                                   j*w_step : j*w_step +self.crop_to[1]]
                    yield image[region]
            
    def gen_img(self):
        '''
        使用生成器方式，首先遍历名称生成器，第二级遍历图像，yield局部
        
        实际上，局部裁切使用tf.extract_image_patches 也可以实现
        
        加入数据的裁切，作为标准化训练手段，也是数据增广
        但是数据裁切就意味着两重生成器，图像内yield和图像的yield更换
        图像内部yield截获异常之后，再更新图像，直到图像遍历完成抛出、传递异常
        '''
        #try:
            #data_p = next(self.ntuple_gen)
        #except GeneratorExit as e:
            #print('所有数据完成遍历')
            #raise GeneratorExit(e)
            ## 传递这个异常，epoch中需要处理这个异常，进行下一轮epoch
        
        #if self.train_mode == 'train':
            ## 重复操作似乎可以map
            ##image_name = os.path.join(self.data_dir,data_p[0])
            ##image = cv2.imread(image_name,-1)            
            ##annot_name = os.path.join(self.data_dir,data_p[1])
            ##annot = cv2.imread(annot_name,-1)
            #image,annot = map(_get_img,[data_p[0],data_p[1]])
            
        #elif self.train_mode == 'test':
            ##image_name = os.path.join(self.data_dir,data_p[0])
            ##image = cv2.imread(image_name,-1)            
            ##annot = None
            #image,annot = map(_get_img,[data_p[0],None])
        #elif self.train_mode == 'predict':
            ##image_name = os.path.join(self.data_dir,data_p[0])
            ##image = cv2.imread(image_name,-1)            
            ##annot = None
            #image,annot = map(_get_img,[data_p[0],None])
        
        #if self.augment == True:
            #image = self._data_augment(image)
            #annot = self._data_augment(annot)
        
        #if self.norm_batch >= 1:
            #'''# 单一图像的归一化  还是作为一个batch进行归一化吧'''
            #image = tf.image.per_image_standardization(image)
        for nameTuple in self.ntuple_gen:
            image,annot = map(self._get_img,nameTuple)
            r_ang = (random.random()-0.5)*8# 正负8度
            image,annot = itl.starmap(self._rotate_bound,[(image,r_ang),(annot,r_ang)])
            # 截取处理
            for image,annot in map(self._gen_crop,(image,annot)):
                # 还是使用这个逻辑实现分离
                yield image,annot
        raise  GeneratorExit(r"所有数据完成一次遍历")

        
        
    
    def _preproc_batch(self, batchData):
        '''
        废弃，使用tf实现
        
        已经组装批次之后
        将批次数据，在yield之前进行预处理，减去均值除以方差。
        方法三种：
        1，最基本的均值方差
        2，白化，数据方差矩阵的svd结果上的投影
        存在的问题，归一化的均值、方差在批次之间的差异，只能是假设这是平稳的。目前很难在全局规模上进行统计
        这个方法仅仅是批次规模的统计
        '''
        if self.norm_batch >= 1:
            #  一个批次数据基本形态是[batch,H,W,channel]
            #基本的均值方差处理
            #得到的均值是一个样本的形状，然后相减的时候自动发生广播
            
            batchData -= np.mean(batchData, axis = 0)
            batchData /= np.std(batchData, axis = 0)
            
            # 适用tensorflow的实现
            # images = tf.map_fn(lambda e: tf.image.per_image_standardization(e),
                                #images,
                                #parallel_iterations=10000)
        elif self.norm_batch==2:
            # 数据ZCA白化 一是特征相关性较低；二是特征具有相同的方差
            # ZCA和PCA白化相比，在于不降维
            # 那么是不是需要降维？！？！？
            # 类似的，在模型中的每一级上需要不需要降维？
            #参见的第一有svd网络、第二有“双线性pooling”
            # 不建议适用
            dataMatrix = np.reshape(batchData,(self.batch_sz,))
            # 将【b,h,w,d】变为【b, size】,留空可以不可以  一个样本一个向量
            corM = np.matmul(dataMatrix,dataMatrix.T)# 自相关矩阵
            U, S, V = np.linalg.svd(corM)# 分解
            dataMatrix = np.dot(dataMatrix, U)
            # 向量恢复图像
            batchData = np.reshape(dataMatrix,batchData.shape)
            
            def zca_whitening_matrix(X):
                """
                Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
                INPUT:  X: [M x N] matrix. 数据已经向量化，一个batch作为一个矩阵
                    Rows: Variables 一个样本是一个列，一行中？？？？
                    Columns: Observations
                OUTPUT: ZCAMatrix: [M x M] matrix 得到的矩阵不是原始形状的
                X = np.array([[0, 2, 2], [1, 1, 0], [2, 0, 1], [1, 3, 5], [10, 10, 10] ]) # Input: X [5 x 3] matrix
                ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
                ZCAMatrix # [5 x 5] matrix
                xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
                xZCAMatrix # [5 x 3] matrix
                """
                # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
                sigma = np.cov(X, rowvar=True) # [M x M]
                # Singular Value Decomposition. X = U * np.diag(S) * V
                U,S,V = np.linalg.svd(sigma)
                    # U: [M x M] eigenvectors of sigma.
                    # S: [M x 1] eigenvalues of sigma.
                    # V: [M x M] transpose of U
                # Whitening constant: prevents division by zero
                epsilon = 1e-5
                # ZCA Whitening matrix: U * Lambda * U'
                ZCAMatrix = np.dot(U, 
                                   np.dot(np.diag(1.0/np.sqrt(S + epsilon)), 
                                          U.T)) # [M x M]
                
                return ZCAMatrix            
            
        return batchData
            
        
        
        
    def get_patchs_gen_slots(self,**dargs):
        ''' 
        废弃，不使用这种方式。遍历太慢
        
        使用多个生成器轮询方式，从多图像得到数据patch
        名称上是patch，实际上是一batch 形状[nbatch, H,W,C]
        这是个生成器
        参数中定义：1，批次大小、2，同时打开图像数目。
        '''
        self.n_imgs = dargs['n_imgs'] if 'n_imgs' in dargs else 16
        # 同时打开图像的数目  ，或是进程池的大小
        print('using %d images '%self.n_imgs)
        
        
        # 图像batch大小。对于局部图像，大小不妨比较大
        
        para_list = [0 ]* self.n_imgs# 一系列生成器管理，list容器，初始为0
        # NOTE:三个初始化数。三种数据是1个单位的样本，将被组装
        # 形状注意减去1。数据类型需要指定否则就是float，但是指定了uint8后，自动变成了int16？？？
        data_batch = np.zeros( (1, self.border_arg*2-1, self.border_arg*2-1, 3),
                                dtype = np.uint8
                            )# 这个重新初始化出了错
        label_batch = [0] #_q = deque(maxlen = batch_sz)
        pos_batch = np.array([[0, 0]],dtype = np.uint8)
        
        n_exhausted = 0 # 图像对子对应的对象耗尽之后的请求次数
        while n_exhausted < self.n_imgs:
            # 当所有槽都发生耗尽，退出
            # 存在问题，最后一个槽的数据（的最后的batch）可能无法有效抛出，但是影响不过一个batch，一般128个左右。不管了
            indexer = random.randint(0, self.n_imgs - 1) # 随机选择一个槽
            if para_list[indexer] == 0 :# 若这个项目是不可用状态，则初始化，装入生成器
                # 这里删去获取局部的操作，免得麻烦
                try:
                    # 所有图像遍历,可能出现耗尽   get_img中含有try能够传递异常
                    img,annot = self.get_img()
                except GeneratorExit as e:
                    print(e)
                    raise GeneratorExit('all data tranversed',e)
                    # 这里不应该break，尚未完成的别的slot将废气
                    n_exhausted += 1 # 一个槽读取完毕，同时所有数据都耗尽，加一
                    continue# 到下一次,略过下面行
                para_list[indexer] = self._patch_frm_a_img( img,
                                                            annot,
                                                            border_arg=self.border_arg)
                # 这个槽装入一个图像到patch的生成器
            else:# 若这个项目有生成器，则取数据。使用try方式防止耗尽，并实现耗尽后标注以重新初始化
                try:
                    patch_img,label, pos = next(para_list[indexer])
                except GeneratorExit:
                    # 这个图像耗尽,，下次访问重新装载新图像
                    para_list[indexer] = 0  
                    continue
                finally:
                    pass

                patch_img = patch_img.reshape(( 1,
                                                patch_img.shape[0],
                                                patch_img.shape[1],
                                                patch_img.shape[2]
                                                )
                                              )
                data_batch = np.concatenate((data_batch,patch_img),0)
                label_batch.append(label)  # 整数不能这么操作
                pos = np.array([pos])# 这个也要转换类型、注意尺寸
                pos_batch = np.concatenate((pos_batch,pos),0)
                # 数据取出：        
                if (data_batch.shape[0] == self.batch_sz)|():
                    # 条件1：检查到满了所以输出、并清空
                    # 条件2：所有槽都耗尽，但是数目不到大小，则抛出剩下的
                    if self.norm_batch:
                        data_batch  = self._preproc_batch(data_batch)
                        label_batch = self._preproc_batch(label_batch)
                        # 位置的相关旋转变换就有问题了
                    yield data_batch,label_batch
                    print('a batch yielded') # 注意这一句是滞后的，是外部执行完毕在此回到生成器才发生的！！！
                    # NOTE: 数据重新初始化, 这里不得不重复
                    data_batch = np.zeros((1, self.border_arg*2-1,self.border_arg*2-1,3),dtype = np.uint8)# 这个重新初始化出了错
                    label_batch = [0] #_q = deque(maxlen = batch_sz)
                    pos_batch = np.array([[0, 0]],dtype = np.uint8)

    def get_patchs_gen_MP(self):
        # TODO: 未实现。
        #       类方法，还是个生成器，能不能、怎么实现多进程？？和静态方法有关么？
        # 一下为了实现生成器的多进程使用，需要定义“多线程代理”对象，但是定义在类内部显然是有问题的，因为必将属于特殊的进程
        # 抽象位置在哪里？？？
        #class GeneratorProxy(BaseProxy):#类定义在这里是不是不合适？？ 这两个类为什么被解析成为“读取图像”函数的内部？？？
            #_exposed_ = ['__next__']
            #def __iter__(self):
                #return self
            #def __next__(self):
                #return self._callmethod('__next__')

        #class MP_Manager(BaseManager):
            ## 一个空的类
            #pass    
        #用不用注册使用内部的？
    
        #MP_Manager.register('mp_patch_frm_a_img', patch_frm_a_img, proxytype=GeneratorProxy)
        
        #manager = MP_Manager()
        #manager.start()
        #mp_patch_frm_a_img = manager.mp_patch_frm_a_img
                    
                #freeze_support()
        #NOTE: 以下是原本的设计原型
        
        
        n_workers = multiprocessing.cpu_count() /2
        p_pool = Pool(processes = n_workers )
        run_q = deque()
        while 1:
            # 若是数据队列满了，则取出数据batch，清空
            #　第一个while
            #    若是当前数据队列未满，则从任务队列中取出数据添加到数据队列
            #  没法出队的时候
            # 
            #  若是任务队列未满，则添加任务
           #  疑问，生成器作为任务，是yield退出还是最后终止退出？！？！
            while len(run_q)>0 and run_q[0].ready():
                # 取出算好数据，并进行返回
                patch_img,a_label = run_q.popleft().get()
                print('got a data')
                # 对于数据不是相同维度的，那么数据将不能以相同的一个ndarray给出
                # 除非使用list，但是后续还是不行
                # 训练模型虽然参数是核心，但是训练中模型尺寸是一致的，不能随着样本变化
                # 而且还有batch内部必须一致的问题
                #data_batch = np.stack((data_batch,patch_img),0)#两个自动增加维度，但是不能累积
                
                label_batch = np.stack((label_batch,a_label),0)# 按照numpy数据组织，【batch,H,W,d】
                if len(data_batch) == batch_sz:
                    yield data_batch,label_batch# 按照numpy数据组织，【batch,H,W,d】
                    data_batch = None
                    label_batch = None
                    print('got a batch')
                if len(run_q)<n_workers:
               # 维护任务队列
                    try:
                   #应该是 patch_frm_a_img运行在独立进程中
                        img,annot = image_reader(data_dir,next(tuple_gen))
                        task = p_pool.map_async(mp_patch_frm_a_img,(img,annot))
                    except Exception:
                        print('error when adding a task ')
                run_q.append(task)
            isQuit = input()
            if isQuit == ord('x'):
                break# 离开while
               #            p_pool.apply_async(task,(next(tuple_gen)))
       
        print('not implemented yet')
        
    




    def get_batch(self):
        '''
        废弃，使用tf.data实现  batch
        
        获取数据的一个batch 图像是完全的，不是局部的
        遍历于依赖名称的生成器，
        获取名称得到文件、得到组装的数据批次都是依赖
        这个生成器内部保留的现场进度
        使用tf.data的from_generator就不需要这个函数了
        '''
        try:
            # 处理循环
            # 数据的大小局限。仅仅适用756,1008,3
            #为了适应可能的别的尺寸的数据，
            #可以采用局部裁切的方式（未实施）
            batch_img_lst = []#np.zeros(self.batch_sz, 756,1008,3)
            batch_ann_lst = []
            for i in range(self.batch_sz):
                img,anno = self.get_img()# 这里会有传递的异常
                # TODO：局部裁切处理
                batch_img_lst.append(img)
                batch_ann_lst.append(anno)
            batch_img  = np.stack(batch_img_lst,axis=0)
            batch_anno = np.stack(batch_ann_lst,axis=0)
            
            batch_img = self._preproc_batch(batch_img)
            batch_anno = self._preproc_batch(batch_anno)
            
            return batch_img,batch_anno
        
        except GeneratorExit:
            raise GeneratorExit# 通知这个batch完成
        
        
'''
数据处理的流水线
已经有了所有图像文件名称、对应标签名称，作为python数据存在于内存

1， 从文件名称、标签对子的序列中进行切片
    Create the dataset from slices of the filenames and labels
2， 重排数据顺序，完全的混乱
    Shuffle the data with a buffer size equal to the length of the dataset. 
    This ensures good shuffling (cf. this answer)
3， 解析文件名称，得到像素数据。可以使用多个线程
    Parse the images from filename to the pixel values. 
    Use multiple threads to improve the speed of preprocessing
4， 可选的，训练中，数据增强，使用多线程加速运行
    (Optional for training) Data augmentation for the images. 
    Use multiple threads to improve the speed of preprocessing
5， 组成、分割batch
    Batch the images
6， 取出一个batch。相当于初始化
    Prefetch one batch to make sure that a batch is ready to be served at all time

def tfdata():
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.map(parse_function, num_parallel_calls=4)
    dataset = dataset.map(train_preprocess, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
def parse_function(filename, label):
    '''
    #读取文件内容
    #解码数据
    #转换到01之间的浮点
    #resize图像
    '''
    image_string = tf.read_file(filename)
    
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [64, 64])
    return resized_image, label    

def train_preprocess(image, label):
    '''
    #训练预处理
    #随机调整亮度、饱和，随机水平翻转'''
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

'''
