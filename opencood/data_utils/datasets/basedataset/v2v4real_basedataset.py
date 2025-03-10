'''
base dataset for v2v4real dataset
modified from RunshengXu's code
by SizheWei
'''
import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import json
import random
import math
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.transformation_utils import dist_two_pose, tfm_to_pose
from opencood.utils.transformation_utils import x1_to_x2_v2v4real as x1_to_x2
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.pcdet_utils.roiaware_pool3d import roiaware_pool3d_utils


class V2V4REALBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        class_names = params.get('class_names', ['Car'])
        self.data_augmentor = DataAugmentor(params['data_augment'], train, params['data_dir'], class_names)

        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir

        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        if 'time_delay' in params:          # number of time delay
            self.tau = params['time_delay']
        else:
            self.tau = 0

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type']  # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center  # will it follows 'self.generate_object_center' when 'self.generate_object_center' change?

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False

        # first load all paths of different scenarios
        scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_folders = scenario_folders

        self.reinitialize()

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                # cav_list = sorted(cav_list)
                random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()
                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.find_camera_files(cav_path,
                                                          timestamp)
                    depth_files = self.find_camera_files(cav_path,
                                                         timestamp, sensor="depth")

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = \
                        camera_files
                    self.scenario_database[i][cav_id][timestamp]['depths'] = \
                        depth_files

                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name

                        # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index

            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                            timestamp_index_delay)
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay

            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(cav_content,
                                                       ego_cav_content,
                                                       timestamp_key,
                                                       timestamp_key_delay,
                                                       cur_ego_pose_flag)
            # load lidar file
            data[cav_id]['lidar_np'] = \
                pcd_utils.pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            data[cav_id]['folder_name'] = \
                cav_content[timestamp_key_delay]['lidar'].split('/')[-3]
            data[cav_id]['index'] = timestamp_index
            data[cav_id]['cav_id'] = int(cav_id)
        return data

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = \
                load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = dist_two_pose(cur_lidar_pose, ego_lidar_pose)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if 'time_delay' in self.params:
            return self.tau
        else:
            if self.async_mode == 'real':
                # noise/time is in ms unit
                overhead_noise = np.random.uniform(0, self.async_overhead)
                tc = self.data_size / self.transmission_speed * 1000
                time_delay = int(overhead_noise + tc + self.backbone_delay)
            elif self.async_mode == 'sim':
                time_delay = np.abs(self.async_overhead)
            else:
                time_delay = 0

            # todo: current 10hz, we may consider 20hz in the future(time delay in ms
            time_delay = time_delay // 100
            return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        if not self.train:
            np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        delay_params['lidar_pose'] = tfm_to_pose(delay_params['lidar_pose'])
        delay_params['true_ego_pos'] = tfm_to_pose(delay_params['true_ego_pos'])

        return delay_params

    @staticmethod
    def find_camera_files(cav_path, timestamp, sensor="camera"):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        sensor : str
            "camera" or "depth"

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + f'_{sensor}3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask, random_seed=None, choice=None, stay_static=False, return_sampled_boxes= False,
                flip=None, rotation=None, scale=None):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'stay_static': stay_static,
                    'return_sampled_boxes': return_sampled_boxes,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        if random_seed:
            self.data_augmentor.random_seed = random_seed
        if choice:
            tmp_dict = self.data_augmentor.forward(tmp_dict, choice)
        else:
            tmp_dict = self.data_augmentor.forward(tmp_dict)
            
        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        if return_sampled_boxes:
            return lidar_np, object_bbx_center, object_bbx_mask, tmp_dict['sampled_gt_boxes']
        return lidar_np, object_bbx_center, object_bbx_mask

    def generate_object_center_lidar(self,
                                     cav_contents,
                                     reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_object_center_v2v4real(cav_contents,
                                                                   reference_lidar_pose)

    def generate_object_center_camera(self,
                                      cav_contents,
                                      reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose
        )

    def get_ext_int(self, params, camera_id):
        camera_coords = np.array(params["camera%d" % camera_id]["cords"]).astype(
            np.float32)
        camera_to_lidar = x1_to_x2(
            camera_coords, params["lidar_pose_clean"]
        ).astype(np.float32)  # T_LiDAR_camera
        camera_to_lidar = camera_to_lidar @ np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32)  # UE4 coord to opencv coord
        camera_intrinsic = np.array(params["camera%d" % camera_id]["intrinsic"]).astype(
            np.float32
        )
        return camera_to_lidar, camera_intrinsic
    
    # xuyunjiang 2025.1 add create gt sampling database
    def create_groundtruth_database(self, used_classes=None, split='train', sensor='vehicle'):
        import torch
        from pathlib import Path
        import pickle
        import tqdm
        from opencood.utils.transformation_utils import x1_to_x2
        from opencood.utils import box_utils

        database_save_path = Path(self.params['data_dir']) / ('gt_database_%s' % sensor) # my_dair-v2x/v2x_c/cooperative-vehicle-infrastructure/gt_database_xxx
        db_info_save_path = Path(self.params['data_dir']) / ('v2v4real_dbinfos_%s.pkl' % sensor)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        print('generating {} gt-databse for augmentation'.format(sensor))

        if sensor == 'mix':
            sensor_list = ['vehicle', 'fusion', 'inf']
        else:
            sensor_list = None

        sample_nums = self.len_record[-1] # 按场景累加
        for k in tqdm.tqdm(range(sample_nums)):
            base_data_dict = self.retrieve_base_data(k)
            sample_idx = k # 样本编号

            if sensor_list is not None:
                sensor = random.choice(sensor_list)

            if sensor == 'vehicle':
                points = base_data_dict[0]['lidar_np']
                annos = base_data_dict[0]['params']['vehicles_single_all'] #ego coord
                boxes_lidar = []
                for anno in annos:
                    x = anno['3d_location']['x']
                    y = anno['3d_location']['y']
                    z = anno['3d_location']['z']
                    l = anno['3d_dimensions']['l']
                    h = anno['3d_dimensions']['h']
                    w = anno['3d_dimensions']['w']
                    rotation = anno['rotation']
                    box_lidar = [x,y,z,l,w,h,rotation]
                    boxes_lidar.append(box_lidar)
            elif sensor == 'fusion':
                # ego_lidar_pose = base_data_dict[0]['params']['lidar_pose'] # 第一个作为ego
                # first find the ego vehicle's lidar pose
                for cav_id, cav_content in base_data_dict.items():
                    if cav_content['ego']:
                        ego_lidar_pose = cav_content['params']['lidar_pose']
                projected_lidar_stack = []
                object_stack = []
                object_id_stack = []
                for cav_id, selected_cav_base in base_data_dict.items():
                    # transformation
                    transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_lidar_pose)
                    lidar_np = selected_cav_base['lidar_np'] # n,4
                    # project the lidar to ego space
                    lidar_np[:, :3] = \
                        box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                    object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base], ego_lidar_pose)
                    object_stack.append(object_bbx_center[object_bbx_mask==1])
                    object_id_stack += object_ids
                    # all these lidar and object coordinates are projected to ego already.
                    projected_lidar_stack.append(lidar_np)
                projected_lidar_stack = np.vstack(projected_lidar_stack) # (n1+n2, 4)
                points = projected_lidar_stack
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack) # (N_duplicate, 7)
                object_stack = object_stack[unique_indices] # (N, 7)

                annos = object_stack # 协同下的label
                boxes_lidar = object_stack
            
            elif sensor == 'inf':
                ego_lidar_pose = base_data_dict[0]['params']['lidar_pose']
                if len(base_data_dict) == 1:
                    continue
                for cav_id, selected_cav_base in base_data_dict.items():
                # transformation
                    if cav_id == 0:
                        continue
                    transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_lidar_pose)
                    lidar_np = selected_cav_base['lidar_np']
                    # project the lidar to ego space
                    lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                    # all these lidar and object coordinates are projected to ego already.
                    points = lidar_np
                    annos = base_data_dict[0]['params']['vehicles_all']
                    boxes_lidar, _ = project_world_objects_dairv2x(annos, ego_lidar_pose)

            elif sensor == 'sensor_mix':
                ego_lidar_pose = base_data_dict[0]['params']['lidar_pose']
                projected_lidar_stack = []
                for cav_id, selected_cav_base in base_data_dict.items():
                    # transformation
                    transformation_matrix = x1_to_x2(selected_cav_base['params']['lidar_pose'], ego_lidar_pose)
                    lidar_np = selected_cav_base['lidar_np']
                    # project the lidar to ego space
                    lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(lidar_np[:, :3], transformation_matrix)
                    # all these lidar and object coordinates are projected to ego already.
                    projected_lidar_stack.append(lidar_np)
                
                #instance-level mixup augmentation
                if len(projected_lidar_stack) == 2:
                    ego_point_num, inf_point_num = projected_lidar_stack[0].shape[0], projected_lidar_stack[1].shape[0]
                    mix_ratio = np.random.random()
                    ego_choose_num = int(ego_point_num * mix_ratio)
                    inf_choose_num = int(ego_point_num * (1-mix_ratio))
                    ego_choose_mask = np.random.choice(ego_point_num, ego_choose_num)
                    inf_choose_mask = np.random.choice(inf_point_num, inf_choose_num)
                    projected_lidar_stack[0] = projected_lidar_stack[0][ego_choose_mask]
                    projected_lidar_stack[1] = projected_lidar_stack[1][inf_choose_mask]


                projected_lidar_stack = np.vstack(projected_lidar_stack)
                points = projected_lidar_stack
                annos = base_data_dict[0]['params']['vehicles_all']
                boxes_lidar, _ = project_world_objects_dairv2x(annos, ego_lidar_pose)

            else:
                raise NameError("gt sampling method: {} is not implementated".format(sensor))
            
            if len(base_data_dict)  == 1 and sensor == 'inf':
                continue
            else:


                names = np.array(['Car' for anno in annos]) # 标签 有各种 ： Car Van Truck Bus 等等
                gt_boxes = np.array(boxes_lidar) # (n, 7)

                num_obj = gt_boxes.shape[0]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu( # (num_points, 3)
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints) # 在gt中的点的索引

                #save early-fusion points
                # filename_pts = '%s_early_fusion_pts.bin' % (sample_idx)
                # filepath = database_save_path / filename_pts
                # with open(filepath, 'w') as f:
                #     points.tofile(f)
                #save gt_boxes
                # filename_gt = '%s_early_fusion_gt' % (sample_idx)
                # filepath = database_save_path / filename_gt
                # np.save(filepath, np.array(corners_lidar_list))

                for i in range(num_obj): # 遍历这个样本中的每个gt
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0] # (n_points, 4) 这个gt中的所有点

                    gt_points[:, :3] -= gt_boxes[i, :3] # 这个gt中的每个点减去这个gt的中心点，相当于平移到以0为中心
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.params['data_dir']))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info) # 如果这个类别已经有了，就追加在后面
                        else:
                            all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def train_parser():
    import argparse
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, default='/public/home/lilingzhi/xyj/CoAlign/opencood/hypes_yaml/v2v4real/lidar_only_with_noise/conquer_single_second_onecycle.yaml',
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt



if __name__ == '__main__':
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print('Dataset Building')
    dairv2x_base_dataset = V2V4REALBaseDataset(hypes, visualize=False, train=True) #[vehicle, inf, cooperative]
    dairv2x_base_dataset.create_groundtruth_database(sensor='fusion') #create gt base for [vehicle, fuison, inf, mix, sensor_mix] data