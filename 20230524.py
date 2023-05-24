import cv2
import numpy as np
import torch
import torchvision
import os

# 加载YOLO模型
def load_yolo_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
    model.load_state_dict(torch.load(weights_path))
    return model

# 图像预处理
def preprocess_image(image, input_size):
    resized_image = cv2.resize(image, input_size)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    resized_image = resized_image.transpose(2, 0, 1)
    resized_image = resized_image.astype(np.float32) / 255.0
    return resized_image

# 执行物体检测
def perform_object_detection(image, model):
    input_size = model.stride.max()  # 使用最大步长的大小
    preprocessed_image = preprocess_image(image, (input_size, input_size))
    input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0)
    detections = model(input_tensor)
    return detections

# 在图像上绘制检测结果
def draw_detections(image, detections):
    results = detections.pandas().xyxy[0]
    for _, result in results.iterrows():
        x1, y1, x2, y2 = int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# 車牌辨識
import pytesseract
def perform_license_plate_recognition(image, detection_results):
    for _, result in detection_results.iterrows():
        plate_image = image[int(result.ymin):int(result.ymax), int(result.xmin):int(result.xmax)]
        plate_text = pytesseract.image_to_string(plate_image, config='--psm 7 --oem 3')
        result['plate_text'] = plate_text.strip()
    return detection_results


# 加载YOLO模型
weights_path = 'yolov5x6.pt'
model = load_yolo_model(weights_path)

# 读取图像
for num in range(,):
    image_path = rf'{num}.jpg'
    image = cv2.imread(image_path)
    input_size = (1280, 1280)
    resized_image = cv2.resize(image, input_size)
    cv2.imwrite(rf'{num}.png', resized_image)
    os.remove(rf'{num}.jpg')





# 执行物体检测
detections = perform_object_detection(image, model)

# 执行车牌识别
recognized_plate = perform_license_plate_recognition(image, detections)

# 绘制检测结果
output_image = draw_detections(image, recognized_plate)

# 显示结果
cv2.imshow('License Plate Recognition', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()










##################################


import  torch
from    torchvision             import transforms
from    torch.utils.data        import DataLoader
from    torch                   import nn, optim
import  os
import  xml.etree.ElementTree   as ET
from    PIL                     import Image


dataset_path    = 'train_dataset'
image_folder    = os.path.join(dataset_path, 'X')           # 這邊是 train_x 資訊
label_folder    = os.path.join(dataset_path, 'Y')           # 這邊是 train_y 資訊
image_files     = os.listdir(image_folder)                  # 我們只需要獲取圖像所有檔名，因為圖像和標籤差異只在副檔名

# for building DataLoader
images = []                                                 # 可以想成這個就是 train_x
labels = []                                                 # 可以想成這個就是 train_y，需要留意的是這裡 train_y 可能會比較多
                                                            # 原因是一張照片裡面可以有多個標籤
for image_file in image_files:                              # image_file = image_files[0]
    image_path  = os.path.join(image_folder, image_file)
    label_file  = image_file.split('.')[0] + '.xml'         # 因為圖片和XML檔名取名一致，所以我們只要換副檔名即可
    label_path  = os.path.join(label_folder, label_file)    # 這樣作業起來比較快速且正確
    image       = Image.open(image_path)
    tree        = ET.parse(label_path)                      # for XML
    root        = tree.getroot()
    objects     = root.findall('object')                    # 抓取XML檔案裡面的類別們
    for obj in objects:
        class_name = obj.find('name').text                  # 獲取該類別名
        bbox = obj.find('bndbox')                           # 獲取框框座標
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        labels.append((xmin, ymin, xmax, ymax, class_name))
    images.append(image)

# 将图像和标签转换为Tensor
transform = transforms.Compose([transforms.ToTensor()])
images = [transform(image) for image in images]

# 打印图像和标签数量
print('图像数量:', len(images))
print('标签数量:', len(labels))

# 创建数据集
dataset = [(image, label) for image, label in zip(images, labels)]

# 创建数据加载器
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 初始化模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0                     
    for images, labels in train_loader:     # type(train_loader)
        # break    # 這行主要適用於開發測試
        # images = images.to(device)
        # labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), 'saved_model.pt')








#######################################################

import xml.etree.ElementTree as ET

xml_file = r'train_dataset/LP/2.xml'

# 解析 XML 檔案
tree = ET.parse(xml_file)
root = tree.getroot()

# 提取目標類別和邊界框信息
object_elem = root.find('object')
class_name = object_elem.find('name').text
bbox_elem = object_elem.find('bndbox')
xmin = int(bbox_elem.find('xmin').text)
ymin = int(bbox_elem.find('ymin').text)
xmax = int(bbox_elem.find('xmax').text)
ymax = int(bbox_elem.find('ymax').text)

# 輸出解析結果
print("Class:", class_name)
print("Bounding Box (xmin, ymin, xmax, ymax):", xmin, ymin, xmax, ymax)





##############################################


