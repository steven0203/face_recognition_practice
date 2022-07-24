import os
import random

from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import torchvision.transforms as transforms



def get_file_list(path):
    dirs=sorted(os.listdir(path))
    X=[]
    y=[]
    for i,identity in enumerate(dirs):
        identity_path=os.path.join(path,identity)
        sample_paths=os.listdir(identity_path)
        for sample_path in sample_paths:
            X.append(os.path.join(identity_path,sample_path))
            y.append(i)
            
    return X,y
    

def data_split(X,y,valid_ratio,seed=1234):
    random.seed(seed)
    idx=[i for i in range(len(X))]
    random.shuffle(idx)
    train_X=[]
    valid_X=[]
    train_y=[]
    valid_y=[]
    for i in idx[int(len(X)*valid_ratio):]:
        train_X.append(X[i])
        train_y.append(y[i])
    for i in idx[:int(len(X)*valid_ratio)]:
        valid_X.append(X[i])
        valid_y.append(y[i])
    return train_X,train_y,valid_X,valid_y


class imgDataset(Dataset):
    def __init__(self, file_list,label_list=None,transform=None,image_size=None):
        self.file_list=file_list
        if label_list is not None:
            self.label_list = torch.LongTensor(label_list)
        self.transform=transform
        self.image_size=image_size
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image=cv2.imread(self.file_list[idx])
        if self.image_size!=None:
            image=cv2.resize(image,self.image_size)
        if self.transform!=None:
            image=self.transform(image)
        if self.label_list!=None:
            return image,self.label_list[idx]
        return image


def detect_face(img,detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(40,40))
    if len(faces)==0:
        return img,None
    box=faces[0]
    (x, y, w, h)=box

    face=img[y:y+h,x:x+w,:]
    return face,box


evaluate_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def caculateFeature(face,model,img_size):
    face=cv2.resize(face,img_size)
    data=evaluate_transform(face)
    data=data.cuda().unsqueeze(0)
    with torch.no_grad():
        result=model(data)
        result=torch.nn.functional.normalize(result)
    return result[0]


