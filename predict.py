from keras.models import load_model
from keras.preprocessing.image import img_to_array,load_img
import json
import numpy as np
import matplotlib.pyplot as plt

# 载入模型
model = load_model('vgg16_model.h5')
print(model)

# label = {'n02086240-ShihTzu': 0, 'n02088364-beagle': 1, 'n02093056-bullterrier': 2, 'n02094433-Yorkshireterrier': 3,
#          'n02097047-schnauzer': 4, 'n02098286-WestHighlandwhiteterrier': 5, 'n02099712-Labradorretriever': 6,
#          'n02102318-cockerspaniel': 7, 'n02105641-OldEnglishsheepdog': 8, 'n02105855-Shetlandsheepdog': 9}

label = {0: 'n02086240-ShihTzu', 1: 'n02088364-beagle', 2: 'n02093056-bullterrier'
         , 3:'n02094433-Yorkshireterrier', 4: 'n02097047-schnauzer', 5: 'n02098286-WestHighlandwhiteterrier',6:'n02099712-Labradorretriever'
         , 7: 'n02102318-cockerspaniel', 8: 'n02105641-OldEnglishsheepdog', 9: 'n02105855-Shetlandsheepdog'}

def predict(image):
    # 导入图片
    image = load_img(image)
    plt.imshow(image)
    image = image.resize((150,150))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,0)

    # The predict_classes method is only available for the Sequential class (which is the class of your first model)
    # but not for the Model class (the class of your second model).
    # With the Model class, you can use the predict method which will give you a vector
    # of probabilities and then get the argmax of this vector (with np.argmax(y_pred1,axis=1)).

    predict = model.predict(image)
    # print(predict)  # [[4.8839979e-15 1.6961620e-11 9.9999988e-01 4.6652686e-12 1.4062473e-12
    #                 # 6.0018976e-08 7.9067742e-12 7.6482122e-14 3.7967412e-14 1.6012470e-12]]
    predict = np.argmax(predict, axis=1)
    # print(predict)  # [2]
    # plt.title(label[str(model.predict_classes(image)[0])])
    plt.title(label[predict[0]])
    plt.axis('off')
    plt.show()

predict('E:/wangyyun_data/07-data/test/n02093056-bullterrier/Niutougeng-is09aa7re.jpg')
predict('E:/wangyyun_data/07-data/test/n02086240-ShihTzu/Chrysanthemum-558o2.jpg')
predict('E:/wangyyun_data/07-data/test/n02088364-beagle/Beagle18439.jpg')
