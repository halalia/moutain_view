'''
用于大量多重山脊线的标注
性能问题，为什么连续均匀拖动鼠标需要的响应太长？？？

'''
import os,sys,shutil,glob,fnmatch
import cv2
import numpy as np
import  functools as ft
##
imagePath = 'D:\_work\python\yuelushan\imagez\large\labled'
##imagePath = 'D:\_work\python\yuelushan\imagez\large'
##saveto_Path = 'D:\_work\python\yuelushan\imagez\large\labeled'
saveto_Path = imagePath


toggleMode = 1 # 显示原始图像还是叠加图像：
# 1 显示标注叠加
# 2 表示原始图像
# 3 表示边缘图像
# 4 表示仅仅是标注图像
anno_size = 2 # 表述时候绘制的半径

# mask、标记都是单一通道的，注意！！！

# canny参数
canny_param = dict(
    canny_hyst_1 = 3,# 参数说明太少，不同于matlab的滤波强度，似乎是连接距离
    canny_hyst_diff = 3,
    canny_hyst_2 = 6,#canny_hyst_1 + canny_hyst_diff,
    # 参数意义解释: 可能边缘的梯度强度，大于上上线，则一定判断为是；小于下限则判断为不是。处于两者之间的，则判断连接性？若是连接到边缘，那么就是边缘；弱连接不到，则不是
    canny_edge = None,
    canny_aperture= 7, # 为什么可用的只有3、5、7，别的即使是奇数也不行？！？！
    cannY_l2grad = 0,
    border_arg = 1,# 边界外推宽度
    border_type = cv2.BORDER_REPLICATE

    )



##
def refresh_img():
    # 利用mask，在图像上标出显示
    # 利用mask，过滤边缘骨架
    # 再第二次将边缘骨架叠加
    global mask_img,showimg,anno_img,img,toggleMode,canny_param
    fine_canny = cv2.Canny( cv2.copyMakeBorder(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_type']),
                           canny_param['canny_hyst_1'],
                           canny_param['canny_hyst_2'],
                           canny_param['canny_edge'],
                           canny_param['canny_aperture'],
                           canny_param['cannY_l2grad'])[ canny_param['border_arg']:-canny_param['border_arg'],
                                                         canny_param['border_arg']:-canny_param['border_arg']
                                                         ]
    if toggleMode==1 :
        # 先叠加找到的mask
        showimg[:,:,0] = np.max(np.dstack((mask_img,img[:,:,0])),
                                axis = 2) # BGR 蓝色
        showimg[:,:,1] = np.min(np.dstack((255-mask_img,img[:,:,1])),axis = 2) 
        showimg[:,:,2] = np.min(np.dstack((255-mask_img,img[:,:,2])),axis = 2) 
        # 获取标注
        anno_img = np.bitwise_and(mask_img,fine_canny)
        # 将标注叠加
        showimg[:,:,0] = np.min(np.dstack((255-anno_img,showimg[:,:,0])),axis = 2) # BGR 蓝色
        showimg[:,:,1] = np.min(np.dstack((255-anno_img,showimg[:,:,1])),axis = 2) 
        showimg[:,:,2] = np.max(np.dstack((anno_img,showimg[:,:,2])),axis = 2)
        cv2.putText(showimg,'annotation',(10,30),cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 0, 0), 1, 2)

    elif toggleMode==2:
        # 仅仅原始图像
        showimg = img.copy()
        cv2.putText(showimg,r'origin',(10,30),cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0, 255), 1, 2)
    elif toggleMode==3:
        # 边缘
        showimg = np.dstack((fine_canny,fine_canny,fine_canny))
        cv2.putText(showimg,r'edges',(10,30),cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0, 255), 1, 2)# 注意在单层图像上怎么写彩色字
    elif toggleMode==4:
        # 仅仅标注
        anno_img = np.bitwise_and(mask_img,fine_canny)
        showimg = np.dstack((anno_img,anno_img,anno_img))
        cv2.putText(showimg,r'label line',(10,30),cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0, 255), 1, 2)

    cv2.putText(showimg,
                ''.join(('line width : ',str(anno_size))),
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0, 255), 1, 2)
    cv2.putText(showimg,
                ''.join(('canny aperture : ',str(canny_param['canny_aperture']))),
                (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0,0, 255), 1, 2)
    cv2.imshow('img',showimg)


## mouse callback function
def draw_circle(event,x,y,flags,param):
    # 作为鼠标相关，参数包括：
    # 1、回调事件接口
    # 2、x
    # 3、y
    # 4、旗标
    # 5、参数  没有使用的文档
    global mask_img#,showimg,anno_img,fine_canny,anno_size
    if flags == cv2.EVENT_FLAG_LBUTTON:# 左键设定
        print("l mouse ")
        cv2.circle(mask_img,(x,y),anno_size,255,-1)
        if event == cv2.EVENT_MOUSEMOVE:
            # 直接使用了参数的位置
            # 怎么作为画笔，即左键按下的状态，又鼠标移动.
            # 使用左键状态、移动事件。
            # 在可见数据上绘制、在模板数据上绘制
            cv2.circle(mask_img,(x,y),anno_size,255,-1)
        refresh_img()


    elif flags == cv2.EVENT_FLAG_RBUTTON:# 右键消除
        print('r mouse down')
        cv2.circle(mask_img,(x,y),anno_size,0,-1)
        if event == cv2.EVENT_MOUSEMOVE: 
            cv2.circle(mask_img,(x,y),anno_size,0,-1)
        refresh_img()



##  

fileNlist= os.listdir(imagePath)
num_files = len(fileNlist)
i = 0
search_mode = 1 # 1递增 -1递减
while(i<num_files):

    a_file = fileNlist[i]
    fullName = os.path.join(imagePath,a_file)
    if ("mask" in a_file)  | ('txt'in a_file) | ('anno' in a_file)|(not os.path.isfile(fullName)):
        # 不能是名字是mask的定义文件、不能是名字是txt的……不能是目录
        # 否则下一个
        i = i + search_mode
        continue

    f_name,f_ext = os.path.splitext(a_file)
    mask_fileN = os.path.join(imagePath,''.join([f_name,'_masker.bmp']))
    anno_fileN = os.path.join(imagePath,''.join([f_name,'_anno.bmp']))

    Save_mask_fileN = os.path.join(saveto_Path,''.join([f_name,'_masker.bmp']))
    Save_anno_fileN = os.path.join(saveto_Path,''.join([f_name,'_anno.bmp']))

    img = cv2.imread(os.path.join(imagePath,a_file),-1)
    fine_canny = cv2.Canny( cv2.copyMakeBorder(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_type']),
                           canny_param['canny_hyst_1'],
                           canny_param['canny_hyst_2'],
                           canny_param['canny_edge'],
                           canny_param['canny_aperture'],
                           canny_param['cannY_l2grad'])[ canny_param['border_arg']:-canny_param['border_arg'],
                                                         canny_param['border_arg']:-canny_param['border_arg']
                                                         ]
    fine_canny_3 = cv2.Canny( cv2.copyMakeBorder(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_type']),
                           canny_param['canny_hyst_1'],
                           canny_param['canny_hyst_2'],
                           canny_param['canny_edge'],
                           3,
                           canny_param['cannY_l2grad'])[ canny_param['border_arg']:-canny_param['border_arg'],
                                                         canny_param['border_arg']:-canny_param['border_arg']
                                                         ]
    fine_canny_7 = cv2.Canny( cv2.copyMakeBorder(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_arg'],
                                                canny_param['border_type']),
                           canny_param['canny_hyst_1'],
                           canny_param['canny_hyst_2'],
                           canny_param['canny_edge'],
                           7,
                           canny_param['cannY_l2grad'])[ canny_param['border_arg']:-canny_param['border_arg'],
                                                         canny_param['border_arg']:-canny_param['border_arg']
                                                         ]
    fine_canny_diff = np.logical_and(fine_canny_3,fine_canny_7)
    if type(img) == type(None):
        # 读取不成功，不会产生异常信号只能检查类型
        i = i + search_mode
        continue

##    cv2.imshow('canny',fine_canny)
    if os.path.isfile(mask_fileN):
        mask_img = cv2.imread(mask_fileN,-1)
    else :
        mask_img = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        
##    if os.path.isfile(anno_fileN):
##        anno_img = cv2.imread(anno_fileN,-1)
##    else :
##        anno_img = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    showimg = img.copy()# 必须复制，不能引用  这个操作有点浪费，好在只有一次
    print('data renew')

    cv2.namedWindow('img')
    cv2.setMouseCallback('img',draw_circle)
##    param = {'useIMG' : img,
##            'useMask' : mask_img,
##            'useAnno' : anno_img
##
##        }
##    cv2.setMouseCallback('img',
##                         ft.partial(draw_circle,**param))

    while(1):# 键盘事件、回调必须while？？
        refresh_img()
        cv2.imshow('img',showimg) # 理论上这里循环中就能够保证循环了阿
        k_stroke = cv2.waitKey()
        if  k_stroke == 27:# esc 退出
            print('exit')
            i = num_files
            cv2.destroyAllWindows()
            break
        # 图像滚动
        elif k_stroke == ord('n'):# N下一个
            print('N pressed, goto next')
            i = i+1# 加一不准确，会有自动家的。
            search_mode = 1
            break
        elif k_stroke == ord('b'):# B上一个
            search_mode = -1
            if i>1:
                print('b pressed, go back')
                i = i- 1# 需要减去更多而不是1，因为有别的形式的文件
            else:
                print('b pressed, already beginning')
            break
        # 保存
        elif k_stroke == ord('w'):
            print('data save ...')
            cv2.imwrite(mask_fileN, mask_img)
            cv2.imwrite(anno_fileN, anno_img)##  数据未完成
##            for fnm in fileNlist:
##                if fnmatch.fnmatch(fnm,''.join((f_name,'.*'))):
##                    shutil.move(os.path.join(imagePath,fnm) ,
##                                os.path.join(saveto_Path ,fnm)
##                                )
            #i = i+1
            print('data saved') 
        # 以下是显示模式控制
        elif k_stroke == ord('p'):# 叠加图
            print('toggle image， overlayed')
            toggleMode = 1
        elif k_stroke == ord('o'):# 切换原图
            print('toggle image，original')
            toggleMode = 2
##            refresh_img()
##            cv2.imshow('img',showimg)
        elif k_stroke == ord('i'):# 边缘
            print('toggle image, edge')
            toggleMode = 3
        elif k_stroke == ord('u'):# 标注边缘
            print('toggle image，annotation')
            toggleMode = 4
            
        # 以下是绘制直径的控制
        elif (k_stroke>48) & (k_stroke <58):
            anno_size = k_stroke-48
            print('labeling with radial = ',anno_size)

        # canny参数控制
        elif k_stroke == ord('q'):
            canny_param['canny_aperture'] = 3
        elif k_stroke == ord('a'):
            canny_param['canny_aperture'] = 5
        elif k_stroke == ord('z'):
            canny_param['canny_aperture'] = 7
    cv2.destroyAllWindows()
        
## []-----------------------------------------
##imagePath = 'D:\_work\python\yuelushan\imagez\large'
##fileNlist= os.listdir(imagePath)
##
##
##def draw_circle(event,x,y,flags,param):
####    print(type(param))
##    if flags == cv2.EVENT_FLAG_LBUTTON:# 左键设定
##        if event == cv2.EVENT_MOUSEMOVE: 
##            # 直接使用了参数的位置
##            # 怎么作为画笔，即左键按下的状态，又鼠标移动.
##            # 使用左键状态、移动事件。
##            # 在可见数据上绘制、在模板数据上绘制
##            cv2.circle(img,(x,y),3,(255,0,0),-1)# 位置变为红色
##
##    if flags == cv2.EVENT_FLAG_RBUTTON:# 右键消除
##        if event == cv2.EVENT_MOUSEMOVE: 
##            cv2.circle(img,(x,y),3,(0,0,255),-1)
##
##
##cv2.namedWindow('')
##cv2.setMouseCallback('',draw_circle)# 如何使用param？？？
##i = 0
##a_file = fileNlist[i]
##fullName = os.path.join(imagePath,a_file)
##if ("mask" in a_file)  | ('txt'in a_file) | ('anno' in a_file)|(not os.path.isfile(fullName)):
##    i = i+1
##
####    else:
####        print(a_file)
##img = cv2.imread(os.path.join(imagePath,a_file),-1)
##
##while(1):# 键盘事件、回调必须while？？
##
##    cv2.imshow('',img)
##    if cv2.waitKey(20)  == 27:
##        cv2.destroyAllWindows()
##        break
##
