import cv2
import numpy as np

def get_mask_contour(mask):

    mask = mask.astype(np.uint8) * 255
    # cv2.imwrite('test.png',mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite("counter.png", img)
    w_idx = np.reshape(contours[0][:,0,0],(1,-1))
    h_idx = np.reshape(contours[0][:,0,1],(1,-1))
    new_mask = np.zeros_like(mask)
    new_mask[h_idx,w_idx] = 1
    cv2.imwrite('test.png', new_mask*255)

    return new_mask

    # box = GD.phrase_GD_box(box, bgr_list[current_frame])

    # point = np.array([[box['center_w'], box['center_h']]], dtype=np.float32)
    # label = np.array([1], np.int32)
    # bbox = np.array([
    #             box['start_w'], box['start_h'], 
    #             box['end_w'], box['end_h']])
    
    # target_loc = (X[camera][box['center_h']][box['center_w']],
    #               Y[camera][box['center_h']][box['center_w']],
    #               Z[camera][box['center_h']][box['center_w']])
    
    # residual = tuple( a - b for a, b in zip(target_loc , current_loc))
    # distance = np.linalg.norm([residual[0], residual[1]])

    # if distance < distance_threshold1:

    # seg, cut, mask = SAM2.inference_image_memory(bgr_list[current_frame],point,label,bbox)   

    # import open3d
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(point_cloud) 
    # pcd_current_loc = open3d.geometry.PointCloud()
    # pcd_current_loc.points = open3d.utility.Vector3dVector(np.array(current_loc).reshape(1,-1))
    # dists = pcd.compute_point_cloud_distance(pcd_current_loc)
    # dists = np.asarray(dists)
    
    # target_loc = (X[camera][box['center_h']][box['center_w']],
    #               Y[camera][box['center_h']][box['center_w']],
    #               Z[camera][box['center_h']][box['center_w']])
    
    # residual = tuple( a - b for a, b in zip(target_loc , current_loc))
    # distance = np.linalg.norm([residual[0], residual[1],residual[2]])

    # for key in targets.keys():
    #     for target_info in targets[key]:
    #         if target_info['end_frame'] != -1:
    #             continue
    #         video_segments = target_info['video_segments']
    #         # checking_saver.save(check,str(current_frame))
    #         for frame in video_segments.keys():
    #             mask = np.squeeze(video_segments[frame][1])
    #             idx = np.where(mask == True)
    #             if box['center_h'] in idx[0] and box['center_w'] in idx[1]:
    #                 return None, None

    # target_info = dict()
    # target_info['start_frame'] = current_frame
    # target_info['end_frame'] = -1

    # point = np.array([[box['center_w'], box['center_h']]], dtype=np.float32)
    # label = np.array([1], np.int32)
    # bbox = np.array([
    #             box['start_w'], box['start_h'], 
    #             box['end_w'], box['end_h']])
    # seg, cut, mask = SAM2.inference_image_memory(bgr_list[current_frame],point,label,bbox)   

    # target_info['point'] = point  
    # target_info['box'] = bbox
    # target_info['label'] = label
    # target_info['point_cloud'] = np.array([[0]])
    # target_info['mask'] = mask
    # target_info['if_target'] = False
    # targets[current_frame].append(target_info)

    # return seg[0], cut[0]

# def update_target(targets, current_frame, bgr_list, SAM2 : SAM2_detector, sam2_saver : image_saver_plt, X, Y, Z , PT : point_cloud, voxel_size = 0.1):

#     for key in targets.keys():
#         for target_info in targets[key]:
#             if target_info['end_frame'] != -1:
#                 continue
#             start_frame = target_info['start_frame']
#             video = bgr_list[start_frame:current_frame+1]
#             point = target_info['point']
#             box = target_info['box']
#             label = np.array([1], np.int32)
#             video_segments, video_rgb, show_func = SAM2.inference_video_memory(video,point,label,box)
#             target_info['video_segments'] = video_segments

#             mask = np.squeeze(video_segments[current_frame-start_frame][1])
#             idx = np.where(mask == True)
#             if idx[0].shape[0] == 0:
#                 target_info['end_frame'] = current_frame
#                 axs = show_func(video_rgb, video_segments)
#                 sam2_saver.save_list(axs,str(start_frame))

#                 # PT.show_point_cloud_go(target_info['point_cloud'][:,0],
#                 #                        target_info['point_cloud'][:,1],
#                 #                        target_info['point_cloud'][:,2])

#                 # z = target_info['point_cloud'][:,2]
#                 # idx = np.where(z<-0.2)[0]
#                 # target_info['point_cloud'] = target_info['point_cloud'][idx,:]

#                 # import open3d
#                 # pcd = open3d.geometry.PointCloud()
#                 # pcd.points = open3d.utility.Vector3dVector(target_info['point_cloud'])
#                 # open3d.visualization.draw_geometries([pcd])
#                 continue  

#             # get contour point cloud
#             # contour_mask = get_mask_contour(mask)            
#             # contour_X, contour_Y, contour_Z = PT.get_point_cloud_from_mask(X,Y,Z,contour_mask)

#             x,y,z = PT.get_point_cloud_from_mask(X,Y,Z,mask)
#             point_cloud = PT.convert_to_open3d(x,y,z)

#             if target_info['point_cloud'].shape[1] == 3:
               
#                point_cloud = np.concatenate((point_cloud,target_info['point_cloud']),axis = 0)
#                import open3d
#                pcd = open3d.geometry.PointCloud()
#                pcd.points = open3d.utility.Vector3dVector(point_cloud)  
#                downsampled = pcd.voxel_down_sample(voxel_size)  
#                points = np.asarray(downsampled.points) 
#                target_info['point_cloud'] = points  

#             else:
#                target_info['point_cloud'] = point_cloud

        # current_point_cloud, current_point_cloud_color = PT.convert_to_open3d_dict(X, Y, Z, bgr)

        # if global_point_cloud == None:
        #     global_point_cloud = current_point_cloud
        #     global_point_cloud_color = current_point_cloud_color
        # else:
        #     global_point_cloud, global_point_cloud_color = PT.combine_point_cloud(global_point_cloud,current_point_cloud,global_point_cloud_color,current_point_cloud_color)

        # if i ==40:
        #     PT.show_point_cloud_numpy(global_point_cloud, global_point_cloud_color)

        # detection = GD.inference_single_image(bgr,'person.')

# def target_point_cloud_check(target_info_1,target_info_2):

            #    origin_mask =  target_info['mask']
            #    origin_mask = origin_mask.astype(np.uint8) * 255
            #    cv2.imwrite('origin_mask.png',origin_mask)

            #    now_mask =  mask
            #    now_mask = now_mask.astype(np.uint8) * 255
            #    cv2.imwrite('now_mask.png',now_mask)               

            #    target_X_origin = target_info['X'].reshape(-1,1)
            #    target_Y_origin = target_info['Y'].reshape(-1,1)
            #    target_Z_origin = target_info['Z'].reshape(-1,1)
            #    xyz_origin = np.concatenate(
            #        (target_X_origin,target_Y_origin,target_Z_origin)
            #     , axis=-1)
               
               
            #    pcd_origin = open3d.geometry.PointCloud()
            #    pcd_origin.points = open3d.utility.Vector3dVector(xyz_origin)

            #    target_x = target_X.reshape(-1,1)
            #    target_y = target_Y.reshape(-1,1)
            #    target_z = target_Z.reshape(-1,1)
            #    xyz = np.concatenate((target_x,target_y,target_z), axis=-1)

            #    pcd = open3d.geometry.PointCloud()
            #    pcd.points = open3d.utility.Vector3dVector(xyz)     

            #    dists = pcd.compute_point_cloud_distance(pcd_origin)
            #    dists = np.asarray(dists) 
            #    inlier_indices = np.where(dists > point_cloud_threshold)[0]
            #    inlier_indices = inlier_indices.reshape(1,-1)
            #    target_info['X'] = np.concatenate(
            #        (target_info['X'],target_X[0][inlier_indices]),axis=-1
            #        )
            #    o=9