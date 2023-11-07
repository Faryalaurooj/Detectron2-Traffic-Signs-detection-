import numpy as np
import os
import shutil
import glob
import argparse
import openpyxl
import pandas as pd

#     return intersection / (box1_area + box2_area - intersection + 1e-6)
class Evaluate:
    def make_alist(self, label_file):
        labels = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                values = line.split(' ')
                cls = int(values[0])
                x = float(values[1])
                y = float(values[2])
                w = float(values[3])
                h = float(values[4])
                labels.append((cls, x,y,w,h))
        return labels

    def calculate_class_wise_yolo_metrics(self, ground_truth_labels, predicted_labels, num_classes=4, iou_threshold=0.5):
        class_metrics = {class_id: {"TP": 0, "FN": 0, "FP": 0} for class_id in range(num_classes)}

        # Initialize a list to track whether each ground truth box has been matched
        gt_matched = [False] * len(ground_truth_labels)

        # Iterate through predicted labels
        for pred_label in predicted_labels:
            pred_class, pred_x, pred_y, pred_width, pred_height = pred_label

            # Find the best matching ground truth box (if any) based on highest IOU
            best_iou = 0
            best_gt_index = -1

            for gt_index, gt_label in enumerate(ground_truth_labels):
                gt_class, gt_x, gt_y, gt_width, gt_height = gt_label

                # Calculate IOU (Intersection over Union)
                x1 = max(pred_x - pred_width / 2, gt_x - gt_width / 2)
                y1 = max(pred_y - pred_height / 2, gt_y - gt_height / 2)
                x2 = min(pred_x + pred_width / 2, gt_x + gt_width / 2)
                y2 = min(pred_y + pred_height / 2, gt_y + gt_height / 2)

                intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
                pred_area = pred_width * pred_height
                gt_area = gt_width * gt_height
                iou = intersection_area / (pred_area + gt_area - intersection_area)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index


            # If a match is found and it exceeds the IOU threshold
            if best_iou >= iou_threshold and pred_class == ground_truth_labels[best_gt_index][0]:
                if not gt_matched[best_gt_index]:
                    class_metrics[pred_class]["TP"] += 1
                    gt_matched[best_gt_index] = True
                else:
                    class_metrics[pred_class]["FP"] += 1
            else:
                class_metrics[pred_class]["FP"] += 1

        # Calculate FN for each class based on unmatched ground truth boxes
        for gt_index, gt_label in enumerate(ground_truth_labels):
            gt_class = gt_label[0]
            if not gt_matched[gt_index]:
                class_metrics[gt_class]["FN"] += 1

        return class_metrics

    def main(self, args):
        Total_TP = 0
        Total_FP = 0
        Total_FN = 0
        classes = {'0':'prohibitory', '1': 'danger', '2': 'mandatory', '3': 'other' }
        df = pd.DataFrame(columns=['image_name', 'TP', 'FP', 'FN'])
        print(args.labels)
        for label_file in glob.glob(args.labels + '*.txt'):
            print(label_file)
            g_labels = self.make_alist(label_file)
            p_label_file = label_file.split('/')[-1]
            name = p_label_file.split('.')[0]
            print(label_file)
            p_labels = self.make_alist(args.preds + name + '.txt')
            class_metrics = self.calculate_class_wise_yolo_metrics(g_labels, p_labels)

            for id, (class_id, metrics) in enumerate(class_metrics.items(), 2):
                print(f"Class {class_id}:")
                print("True Positives:", metrics["TP"])
                print("False Negatives:", metrics["FN"])
                print("False Positives:", metrics["FP"])
                TP = metrics['TP']
                FP = metrics['FP']
                FN = metrics['FN']
                # new_row = {'image_name': name,
                #            'Class-id': classes[str(class_id)],
                #            'TP': TP,
                #            'FP': FP,
                #            'FN': FN,
                #            }

                # df = df.append(new_row, ignore_index=True)
                Total_TP = Total_TP + TP
                Total_FP = Total_FP + FP
                Total_FN = Total_FN + FN
        Precision = Total_TP / (Total_TP + Total_FP)
        Recall = Total_TP / (Total_TP + Total_FN)
        new_row = {'image_name':'Total' ,
                   'Class-id': 'All Images',
                   'TP': Total_TP,
                   'FP': Total_FP,
                   'FN': Total_FN,
                   "Precision": Precision,
                   'Recall': Recall,
                   'F1': (2*Precision*Recall) / (Recall + Precision)

                   }
        print(new_row)
        # df = df.append(new_row, ignore_index = True)
        # df.to_excel(args.final_result + 'evaluation_result.xlsx', index=False)

# if __name__== '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--labels', type=str, default='evaluation/labels/', help='path to the ground truth')
#     parser.add_argument('--preds', type=str, default='evaluation/predicted/', help='path to the predicted labels')
#     parser.add_argument('--result', type=str, default='evaluate/', help='evaluation result file path')
#     args = parser.parse_args()
#     main(args)
#     # ground_truth_labels = [(0, 0.5, 0.5, 0.4, 0.4), (1, 0.7, 0.7, 0.3, 0.3)]
#     # predicted_labels = [(0, 0.55, 0.55, 0.35, 0.35), (2, 0.2, 0.2, 0.3, 0.3)]
#     # iou_threshold = 0.5
#     #workbook = openpyxl.Workbook()
#     #sheet = workbook.active





