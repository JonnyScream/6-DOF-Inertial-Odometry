import argparse
import numpy as np
import os

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# from sklearn.externals import joblib

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['oxiod', 'euroc'], help='Training dataset name (\'oxiod\' or \'euroc\')')
    parser.add_argument('--model', help='Model path')
    parser.add_argument('--dataset_path', help='Model path')
    args = parser.parse_args()

    model = load_model(args.model, custom_objects={'CustomMultiLossLayer': CustomMultiLossLayer})

    window_size = 200
    stride = 10

    imu_data_filenames = []
    gt_data_filenames = []

    if args.dataset == 'oxiod':
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu2.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu5.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/imu6.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu1.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/imu3.csv')
        imu_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/imu1.csv')

        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi2.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi5.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data1/syn/vi6.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data3/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi1.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data4/syn/vi3.csv')
        gt_data_filenames.append('Oxford Inertial Odometry Dataset/handheld/data5/syn/vi1.csv')

    elif args.dataset == 'euroc':
        imu_data_filenames.append('MH_02_easy/mav0/imu0/data.csv')
        imu_data_filenames.append('MH_04_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_03_difficult/mav0/imu0/data.csv')
        imu_data_filenames.append('V2_02_medium/mav0/imu0/data.csv')
        imu_data_filenames.append('V1_01_easy/mav0/imu0/data.csv')

        gt_data_filenames.append('MH_02_easy/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('MH_04_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V1_03_difficult/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V2_02_medium/mav0/state_groundtruth_estimate0/data.csv')
        gt_data_filenames.append('V1_01_easy/mav0/state_groundtruth_estimate0/data.csv')

    imu_data_filenames_full = [os.path.join(args.dataset_path, file_name) for file_name in imu_data_filenames]
    gt_data_filenames_full = [os.path.join(args.dataset_path, file_name) for file_name in gt_data_filenames]

    for (cur_imu_data_filename_full, cur_gt_data_filename_full, cur_imu_data_filename) in zip(imu_data_filenames_full, gt_data_filenames_full, imu_data_filenames):
        output_dir = './plots_folder'
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, cur_imu_data_filename.split('/')[0] + ".html")

        if args.dataset == 'oxiod':
            gyro_data, acc_data, pos_data, ori_data = load_oxiod_dataset(cur_imu_data_filename_full, cur_gt_data_filename_full)
        elif args.dataset == 'euroc':
            gyro_data, acc_data, pos_data, ori_data = load_euroc_mav_dataset(cur_imu_data_filename_full, cur_gt_data_filename_full)

        [x_gyro, x_acc], [y_delta_p, y_delta_q], init_p, init_q = load_dataset_6d_quat(gyro_data, acc_data, pos_data, ori_data, window_size, stride)
        
        if args.dataset == 'oxiod':
            [yhat_delta_p, yhat_delta_q] = model.predict([x_gyro[0:200, :, :], x_acc[0:200, :, :]], batch_size=1, verbose=0)
        elif args.dataset == 'euroc':
            dummy_y_delta_p = np.zeros_like(x_gyro)[:,0]
            dummy_y_delta_q = np.zeros([x_acc.shape[0],4])
            out = model.predict([x_gyro, x_acc,dummy_y_delta_p,dummy_y_delta_q], batch_size=1, verbose=0)

        yhat_delta_p, yhat_delta_q = out[:,7:10], out[:,10:]
        gt_trajectory = generate_trajectory_6d_quat(init_p, init_q, y_delta_p, y_delta_q)
        pred_trajectory = generate_trajectory_6d_quat(init_p, init_q, yhat_delta_p, yhat_delta_q)

        if args.dataset == 'oxiod':
            pred_trajectory = pred_trajectory[0:200, :]
            gt_trajectory = gt_trajectory[0:200, :]

        trajectory_rmse = np.sqrt(np.mean(np.square(np.linalg.norm(pred_trajectory - gt_trajectory, axis=-1))))
        print('Trajectory RMSE, sequence %s: %f' % (cur_imu_data_filename.split('/')[0], trajectory_rmse))
        plot_traj(gt_trajectory, pred_trajectory, fig_path, cur_imu_data_filename.split('/')[0])

if __name__ == '__main__':
    main()