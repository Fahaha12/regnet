import argparse
import torch
from model.SwinTransformerForRegression import SwinTransformerForRegression
from model.MultiScaleSwinTransformerForRegression2 import MultiScaleSwinTransformerForRegression
from torchvision import transforms
from PIL import Image
import os
import xml.etree.ElementTree as ET
import csv

def predict_and_save(image_dir, model_path, output_csv, model_type):
    """
    预测指定目录下所有图像，并将结果保存到 CSV 文件。

    Args:
        image_dir (str): 包含图像的目录路径。
        model_path (str): 模型文件路径。
        output_csv (str): 输出 CSV 文件路径。
        model_type (str): 模型类型，可选值为 'swin_transformer' 或 'msstr'。
    """
    # 加载训练好的模型
    if model_type == 'swin_transformer':
        model = SwinTransformerForRegression()
    elif model_type == 'msstr':
        model = MultiScaleSwinTransformerForRegression()
    else:
        raise ValueError("Unsupported model type. Choose 'swin_transformer' or 'msstr'.")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义图像预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder Name', 'Predicted Value'])

        # 遍历所有子文件夹
        for folder_name in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            predicted_values = []

            # 遍历当前文件夹下的所有 JPG 文件
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    image_path = os.path.join(folder_path, filename)
                    xml_path = os.path.splitext(image_path)[0] + ".xml"

                    # 读取图像
                    image = Image.open(image_path).convert('RGB')
                    image = transform(image).unsqueeze(0)

                    # 从XML文件读取表型信息
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    characteristics_element = root.find(".//characteristics")

                    # 从子标签中提取特征值，排除 Survival_days
                    characteristics = []
                    for child in characteristics_element:
                        tag_name = child.tag
                        if tag_name == "Survival_days":  # 跳过 Survival_days
                            continue
                        tag_value = child.text
                        # 处理 "NA" 值
                        if tag_value == "NA":
                            tag_value = 0
                        else:
                            tag_value = float(tag_value)
                        characteristics.append(tag_value)

                    # 假设 characteristics 现在包含 8 个值
                    phenotype = torch.tensor([characteristics], dtype=torch.float32)

                    # 进行预测
                    with torch.no_grad():
                        image = image.to(device)  # 将 image 移动到 GPU
                        phenotype = phenotype.to(device)  # 将 phenotype 也移动到 GPU
                        output = model(image, phenotype)
                        predicted_value = output.item()

                    predicted_values.append(predicted_value)

            # 计算当前文件夹的平均预测值
            if predicted_values:
                average_value = sum(predicted_values) / len(predicted_values)
                csv_writer.writerow([folder_name, f'{average_value:.4f}'])

def main():
    parser = argparse.ArgumentParser(description='Validate the trained model')
    parser.add_argument('--image-dir', type=str, required=True, default='./dataset/BraTS2019_val',
                        help='path to the directory containing all the folders with input images')
    parser.add_argument('--model-path', type=str, required=True, default='./result/latest_model_checkpoint_epoch_500.pth', help='path to the trained model checkpoint')
    parser.add_argument('--output-csv', type=str, default='./result/predictions.csv',
                        help='path to the output CSV file')
    parser.add_argument('--model-type', type=str, default='msstr', choices=['swin_transformer', 'msstr'],
                        help='the type of model used')
    args = parser.parse_args()

    predict_and_save(args.image_dir, args.model_path, args.output_csv, args.model_type)

if __name__ == '__main__':
    main()
