import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
#alex
import cv2
import matplotlib.pyplot as plt
import os
import sys
import multiprocessing as mp

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
# images = np.array(
#     [ia.quokka(size=(64, 64)) for _ in range(32)],
#     dtype=np.uint8
# )



def augmentateImg(fromPathChild,toPathChild,minimal=1000):
    #读取path的jpg文件
    # todo 如果不足 minimal,随机抽取病复制现有的图片，去补足
    list = os.listdir(fromPathChild)
    # print(list)
    images = []
    for f in list:
        imgFile = os.path.join(fromPathChild,f)
        temp = cv2.imread(imgFile)
        images.append(temp)
    print(len(images))

    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),    # vertical flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq.augment_images(images)
    print(len(images_aug))
    name = 0
    for img in images_aug:
        name = name + 1
        # plt.imshow(img)
        # plt.show()
        cv2.imwrite( os.path.join(toPathChild,str(name) + ".jpg"), img)


def multiProc(processor, fromchildPathes,tochildPathes):
    # mp.cpu_count()
    # for childFolder in childFolders:
    #     p = mp.Process(target=reformImgWithROI, args=(childFolder,))
    #     p.start()
    p = mp.Pool(processes=mp.cpu_count())
    for i in range(len(fromchildPathes)):
        fromPathChild = fromchildPathes[i]
        toPathChild = tochildPathes[i]
        print( (fromPathChild,toPathChild) )
        r = p.apply_async(processor, (fromPathChild,toPathChild,))

    p.close()
    p.join()
if __name__ == "__main__" :
    print("--->>>begin augmentation")
    fromchildPathes = []
    tochildPathes = []
    fromPath = sys.argv[1]
    fromList = os.listdir(fromPath)# from
    toPath = sys.argv[2]# to
    # 建立生成目录
    for child in fromList:
        p1 =  os.path.join(toPath,child)
        os.makedirs(p1,exist_ok=True)

    # 处理数据
    for child in fromList:
        fromPathChild = os.path.join(fromPath,child)
        toPathChild = os.path.join(toPath, child)
        if os.path.isdir(toPathChild):
            fromchildPathes.append(fromPathChild)
            tochildPathes.append(toPathChild)
    print("--->>>begin multiProc")
    multiProc(augmentateImg,fromchildPathes,tochildPathes )