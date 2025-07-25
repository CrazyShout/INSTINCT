# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


from matplotlib import pyplot as plt
import numpy as np
import copy
import matplotlib

from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev

# COLOR = ['DeepPink','Teal','GoldenRod','RoyalBlue','MediumPurple']
COLOR = ['White','White','White','White','White']
# COLOR = ['Navy','Navy','Navy','Navy','DeepPink']
COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
# COLOR_RGB[0] = tuple([102, 8, 116])
# COLOR_RGB[1] = tuple([0, 168, 150])
# COLOR_RGB[2] = tuple([255, 143, 0])
# COLOR_RGB[3] = tuple([63, 81, 181])
# COLOR_RGB[4] = tuple([139, 195, 74])

COLOR_RGB[0] = tuple([0, 191, 255])
# COLOR_RGB[1] = tuple([0, 191, 255])
COLOR_RGB[1] = tuple([255, 100, 0])
COLOR_RGB[2] = tuple([0, 191, 255])
COLOR_RGB[3] = tuple([0, 191, 255])
COLOR_RGB[4] = tuple([0, 191, 255])


# COLOR_RGB = [ tuple([int(cc * 255) for cc in matplotlib.colors.to_rgb(c)]) for c in COLOR]
COLOR_PC = [tuple([int(cc*0.4 + 255*0.6) for cc in c]) for c in COLOR_RGB]

def visualize(infer_result, pcd, pc_range, save_path, method='3d', vis_gt_box=True, vis_pred_box=True, left_hand=False, pcd_agent_split=None):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax] Dair-v2x是[-100.8, -40, -3, 100.8, 40, 1]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/40, (pc_range[4]-pc_range[1])/40]) # 201.6/40， 80/40
        pc_range = [int(i) for i in pc_range] # 转换成整型 （-100， -40， -3， 100， 40, 1）
        pcd_np = pcd.cpu().numpy() # cpu上，转为numpy
        split_points_index = np.cumsum(pcd_agent_split)[:-1] # 同场景点云划分
        agent_split_pcd = np.split(pcd_np, split_points_index)

        pred_box_tensor = infer_result.get("pred_box_tensor", None) # N, 8, 3
        gt_box_tensor = infer_result.get("gt_box_tensor", None) # N, 8, 3

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy() # N, 8, 3
            pred_name = ['pred'] * pred_box_np.shape[0] # ['pred', 'pred', 'pred'...] N个 

            score = infer_result.get("score_tensor", None) # （N，）
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])] # ['score: x.xx'...]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                # uncertainty_np = np.exp(uncertainty_np) 不需要再去求指数了，因为已经在前面做过处理
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting. 特殊欧式群？xyj 2024年05月23日

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = gt_box_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            '''
            # input:
                canvas_shape: (800, 2000)
                canvas_x_range: (-100, 100)
                canvas_y_range: (-40, 40)
                left_hand: dair-v2x: True / v2xset:False / opv2v:False
            '''
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*10, (pc_range[3]-pc_range[0])*10),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            # canvas_bg_color=(242,242,242), # 灰色背景
                                            canvas_bg_color=(0,0,0), # 灰色背景
                                            left_hand=left_hand) 
            # canvas_xy（N, 2）, valid_mask (N, ) 在lidar范围内的才标记的掩码
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords 将点云的x，y单独取出，映射到canvas大小上
            # canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points 这是将点云画出来，可以自己选择颜色
            for i, agent_pcd in enumerate(agent_split_pcd):
                canvas_xy, valid_mask = canvas.get_canvas_coords(agent_pcd) # Get Canvas Coords
                agent_colors = COLOR_RGB[i]
                canvas.draw_canvas_points(canvas_xy[valid_mask], colors=agent_colors) # Only draw valid points

            if gt_box_tensor is not None:
                # canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name) # 画绿色GT bbx
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=None) # 画绿色GT bbx
            if pred_box_tensor is not None:
                # canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name) # 画红色GT bbx
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=None) # 画红色GT bbx

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)



        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(canvas_bg_color=(0,0,0), left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
            # canvas.draw_canvas_points(canvas_xy[valid_mask])
            for i, agent_pcd in enumerate(agent_split_pcd):
                canvas_xy, valid_mask = canvas.get_canvas_coords(agent_pcd) # Get Canvas Coords
                agent_colors = COLOR_RGB[i]
                canvas.draw_canvas_points(canvas_xy[valid_mask], colors=agent_colors) # Only draw valid points

            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=None)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=None)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=500)
        plt.clf()
        plt.close()


