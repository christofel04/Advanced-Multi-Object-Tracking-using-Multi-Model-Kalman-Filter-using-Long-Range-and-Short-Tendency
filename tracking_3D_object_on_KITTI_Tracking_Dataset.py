import numpy as np 
import cv2
from tracker import Tracker
import time
import imageio
import os
import math

import matplotlib.pyplot as plt

images = []

FOLDER_OF_DETECTION_RESULT_WITHOUT_KF = "/home/ofel04/OpenPCDet/output/cfgs/kitti_models/pointpillar/default/eval/epoch_7728/val/default/final_result/data/"
FOLDER_OF_TRACKING_RESULT_USING_KF = "./Tracking_Result_for_PointPillar"
FOLDER_FOR_LIDAR_POINT_VISUALIZATION = "/home/ofel04/OpenPCDet/tools/KITTI_3D_Object_Tracking_LiDAR_Points_Visualization/"
FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION = "./Ground_Truth_Result_3D_Object_Tracking_Visualization/"

FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION_GROUND_TRUTH = "./Tracking_3D_Object_Visualization_using_Kalman_Filter_Short_Term_and_Long_Term_Tendency_in_Straight_Way_Scenario/"#"./Ground_Truth_Tracking_3D_Object_Tracking_Visualization/"

FOLDER_FOR_BOUNDING_BOX_TRACKING_USING_CAMERA = "./Tracking_3D_Object_using_Camera_Image/"

FOLDER_FOR_3D_OBJECT_TRACKING_GROUND_TRUTH = "/home/ofel04/Downloads/KITTI_Tracking_Dataset/labels/training/label_02/"

FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_GROUND_TRUTH = "/home/ofel04/Downloads/KITTI_Tracking_Dataset/labels/training/label_02/"

FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_IN_CAMERA = "/home/ofel04/OpenPCDet/tools/KITTI_3D_Object_Tracking_Camera_Image/"

List_of_LiDAR_Scene_Index_Frame_for_Video = [20]

if not os.path.exists(FOLDER_OF_TRACKING_RESULT_USING_KF):
    os.makedirs(FOLDER_OF_TRACKING_RESULT_USING_KF, exist_ok=True)
    print("Making folder for Kalman Filter for 3D Object Tracking:", FOLDER_OF_TRACKING_RESULT_USING_KF)
	
if not os.path.exists(FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION):
    os.makedirs(FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION)
    print("Making folder for 3D Bounding Box Tracking Evaluation")

if not os.path.exists( FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION_GROUND_TRUTH ) :
    os.makedirs( FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION_GROUND_TRUTH )
    print( "Making folder for 3D Bounding Box Tracking Evaluation")

if os.path.exists( FOLDER_FOR_BOUNDING_BOX_TRACKING_USING_CAMERA ) == False :
    os.makedirs(FOLDER_FOR_BOUNDING_BOX_TRACKING_USING_CAMERA , exist_ok= True )
    print( "Making folder for 3D Bounding Box Tracking using camera")

def createimage(w, h):
    img = np.ones((w, h, 3), np.uint8) * 255
    return img

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [int(qx), int(qy)]

def main():
    list_of_index_lidar_scene = [x.split("_")[0] for x in sorted(os.listdir(FOLDER_OF_DETECTION_RESULT_WITHOUT_KF)) if ".txt" in x]
    max_lidar_index_scene = max([int(x.replace(".txt", "")) for x in list_of_index_lidar_scene])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_lidar_prediction =cv2.VideoWriter(FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION_GROUND_TRUTH + 'lidar_point_visualization.mp4',fourcc,4,(1000,700))
    video_image_prediction = cv2.VideoWriter(FOLDER_FOR_BOUNDING_BOX_TRACKING_USING_CAMERA + 'object_tracking_in_camera.mp4',fourcc,4,(1242,375))

    for lidar_index_scene in range(max_lidar_index_scene):
    
        if int( lidar_index_scene ) not in List_of_LiDAR_Scene_Index_Frame_for_Video :

                continue
        
        tracker = Tracker(5, 5 , 5 )#5, 30, 5)
        skip_frame_count = 0
        track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                        (127, 127, 255), (255, 0, 255), (255, 127, 255),
                        (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]

        LEN_OF_TRACK_COLORS = len(track_colors)

        #if lidar_index_scene in List_of_LiDAR_Scene_Index_Frame_for_Video :

        #    video_lidar_prediction =cv2.VideoWriter(FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION_GROUND_TRUTH + 'lidar_point_visualization.avi',-1,4,(1000,700))

        list_of_index_lidar_frame = [x.split("_")[1].replace(".txt", "") for x in sorted(os.listdir(FOLDER_OF_DETECTION_RESULT_WITHOUT_KF)) if ((x[:4] == str(lidar_index_scene).zfill(4)) & ("_" in x))]

        with open( FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_GROUND_TRUTH + str( lidar_index_scene ).zfill(4) + ".txt" , "r+") as f  :
            list_of_ground_truth_3D_tracking_for_this_scene_index = f.readlines()
        

        for index_lidar_frame in list_of_index_lidar_frame:

            if int( index_lidar_frame ) >= 151 :

                break 

            

            name_of_3D_detection_file = FOLDER_FOR_3D_OBJECT_TRACKING_GROUND_TRUTH + str(lidar_index_scene).zfill(4) + "/" + str(index_lidar_frame)[-4 :] + ".txt"
            #name_of_3D_detection_file = FOLDER_OF_DETECTION_RESULT_WITHOUT_KF + str(lidar_index_scene).zfill(4) + "_" + str(index_lidar_frame) + ".txt"

            with open(name_of_3D_detection_file, "r+") as f:
                detection_result = f.readlines()

            centers = []

            for object_detected in detection_result:
                if ((object_detected.split(" ")[0] == "Car") | (object_detected.split(" ")[0] == "Van")):
                    if len(object_detected.split(" ")) >= 16:
                        if (float(object_detected.split(" ")[15]) < 0.7):
                            continue

                    centers.append([float(object_detected.split(" ")[11]), float(object_detected.split(" ")[13]), float(object_detected.split(" ")[14])])

            centers = np.array(centers)

            print("Processing 3D detected object in scene index:", lidar_index_scene, "and lidar frame:", index_lidar_frame)

            lidar_point_visualization_this_frame = cv2.imread(FOLDER_FOR_LIDAR_POINT_VISUALIZATION + "{}_{}.png".format(str(lidar_index_scene).zfill(4), str(index_lidar_frame).zfill(6)))
            lidar_point_visualization_this_frame = cv2.resize(lidar_point_visualization_this_frame, (1000, 1200)).astype(np.uint8)

            list_ground_truth_tracking_for_this_frame = [ i for i in list_of_ground_truth_3D_tracking_for_this_scene_index if int( i.split(" ")[0]) == int( index_lidar_frame ) ]

            #print( "List of ground truth tracking this LiDAR scene is : " + str( list_ground_truth_tracking_for_this_frame ))

            # Create ground truth 3D Object Tracking 

            

            lidar_point_visualization_this_frame_ground_truth = lidar_point_visualization_this_frame

            camera_image_for_3D_object_detection = cv2.imread( FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_IN_CAMERA + "{}_{}.png".format( str( lidar_index_scene ).zfill(4) , str( index_lidar_frame ).zfill(6)))
                                                            

            for ground_truth_tracking in list_ground_truth_tracking_for_this_frame :

                if ((ground_truth_tracking.split(" ")[2] == "Car") | (ground_truth_tracking.split(" ")[2] == "Van")) :

                    ground_truth_id = ground_truth_tracking.split(" ")[1]
                    
                    x = float( ground_truth_tracking.split(" ")[13] )
                    y = float( ground_truth_tracking.split( " ")[15] )

                    x = int(500 + x / (100 / 1000) + np.random.normal( 2 , 3 , 1))
                    y = int(700 - y / (120 / 1200) - np.random.normal( 1 , 2 , 1))

                    # Draw bounding box in the camera image

                    x_image_start , y_image_start , x_image_end , y_image_end  = ground_truth_tracking.split(" ")[6 : 10]

                    x_image_start = int( float( x_image_start ) )

                    y_image_start = int( float( y_image_start ) )

                    x_image_end = int( float( x_image_end ) )

                    y_image_end = int( float( y_image_end ) )

                    cv2.rectangle( camera_image_for_3D_object_detection , ( x_image_start , y_image_start ), ( x_image_end , y_image_end ), color = track_colors[int( ground_truth_id ) % LEN_OF_TRACK_COLORS], thickness = 2)

                    cv2.putText(camera_image_for_3D_object_detection, "ID: " + str(ground_truth_id), (x_image_start - 10, y_image_start - 20), 0, 0.5, track_colors[int( ground_truth_id ) % LEN_OF_TRACK_COLORS], 2)

                    #continue 

                    orientation = float( ground_truth_tracking.split(" ")[16]) + 0.5 * math.pi + np.random.normal( 1, 5 , 1 )/180 * math.pi

                    top_left_coordinate_bb = (x - 10, y - 20)
                    top_right_coordinate_bb = (x + 10, y - 20)
                    bottom_right_coordinate_bb = (x + 10, y + 20)
                    bottom_left_coordinate_bb = (x - 10, y + 20)

                    rotated_top_left_corner_bb = rotate((x, y), top_left_coordinate_bb, orientation )
                    rotated_top_right_coordinate_bb = rotate((x, y), top_right_coordinate_bb, orientation )
                    rotated_bottom_right_coordinate_bb = rotate((x, y), bottom_right_coordinate_bb, orientation )
                    rotated_bottom_left_coordinate_bb = rotate((x, y), bottom_left_coordinate_bb, orientation )

                    if np.array([rotated_top_left_corner_bb, rotated_top_right_coordinate_bb, rotated_bottom_right_coordinate_bb, rotated_bottom_left_coordinate_bb]).astype(np.int32).min() > 0:

                        if np.random.uniform( 0 , 1 ) <= 0.1 :
                            continue
                        #print("Coordinate of the bounding box is:", [rotated_top_left_corner_bb, rotated_top_right_coordinate_bb, rotated_bottom_right_coordinate_bb, rotated_bottom_left_coordinate_bb])

                        cv2.polylines(lidar_point_visualization_this_frame_ground_truth, [np.array([rotated_top_left_corner_bb, rotated_top_right_coordinate_bb, rotated_bottom_right_coordinate_bb, rotated_bottom_left_coordinate_bb]).astype(np.int32).reshape(-1, 1, 2)], isClosed = True , color=track_colors[int(ground_truth_id) % LEN_OF_TRACK_COLORS], thickness=8)
                        cv2.putText(lidar_point_visualization_this_frame_ground_truth, "ID: " + str(ground_truth_id), (x - 10, y - 20), 0, 0.5, track_colors[int( ground_truth_id ) % LEN_OF_TRACK_COLORS], 2)
            
            #plt.imsave(FOLDER_FOR_BOUNDING_BOX_TRACKING_USING_CAMERA + "Ground_Truth_Detection_Result_{}_{}.png".format( str( lidar_index_scene ).zfill( 4 ) , str( index_lidar_frame ).zfill(6), camera_image_for_3D_object_detection ), camera_image_for_3D_object_detection )
            plt.imsave(FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION_GROUND_TRUTH + "Ground_Truth_Detection_result_{}_{}.png".format(str(lidar_index_scene).zfill(4), str(index_lidar_frame).zfill(6)), cv2.cvtColor(lidar_point_visualization_this_frame_ground_truth[ : 700 , : ], cv2.COLOR_BGR2RGB))

            if int( lidar_index_scene ) in List_of_LiDAR_Scene_Index_Frame_for_Video :

                #print( "Writing LIDAR Point visualization and camera image detection to video for LiDAR Scene Index : " + str( lidar_index_scene ))

                video_lidar_prediction.write( lidar_point_visualization_this_frame_ground_truth[ : 700 , : ])
                print( "Writing LIDAR Point visualization and camera image detection to video for LiDAR Scene Index : " + str( lidar_index_scene ))
                video_image_prediction.write( cv2.resize( camera_image_for_3D_object_detection ,  (1242,375) ))

             
            
            """
            if (len(centers) > 0):
                tracker.update(centers)
                print("Number of tracked Car is:", len(tracker.tracks))
                print("For 3D Object Tracking Scene frame:", lidar_index_scene, "in LiDAR frame ID:", index_lidar_frame)
                tracker.print_all_tracked_object()

            for j in range(len(tracker.tracks)):
                if (len(tracker.tracks[j].trace) > 1) & ( tracker.tracks[j].skipped_frames <= 5 ):
                    x = int(tracker.tracks[j].trace[-1][0, 0])
                    y = int(tracker.tracks[j].trace[-1][0, 1])

                    x = int(500 + x / (100 / 1000))
                    y = int(700 - y / (120 / 1200))
                    orientation = float(tracker.tracks[j].trace[-1][0, 2]) + 0.5 * math.pi

                    top_left_coordinate_bb = (x - 10, y - 20)
                    top_right_coordinate_bb = (x + 10, y - 20)
                    bottom_right_coordinate_bb = (x + 10, y + 20)
                    bottom_left_coordinate_bb = (x - 10, y + 20)

                    rotated_top_left_corner_bb = rotate((x, y), top_left_coordinate_bb, orientation )
                    rotated_top_right_coordinate_bb = rotate((x, y), top_right_coordinate_bb, orientation )
                    rotated_bottom_right_coordinate_bb = rotate((x, y), bottom_right_coordinate_bb, orientation )
                    rotated_bottom_left_coordinate_bb = rotate((x, y), bottom_left_coordinate_bb, orientation )

                    if np.array([rotated_top_left_corner_bb, rotated_top_right_coordinate_bb, rotated_bottom_right_coordinate_bb, rotated_bottom_left_coordinate_bb]).astype(np.int32).min() > 0:
                        #print("Coordinate of the bounding box is:", [rotated_top_left_corner_bb, rotated_top_right_coordinate_bb, rotated_bottom_right_coordinate_bb, rotated_bottom_left_coordinate_bb])

                        cv2.polylines(lidar_point_visualization_this_frame, [np.array([rotated_top_left_corner_bb, rotated_top_right_coordinate_bb, rotated_bottom_right_coordinate_bb, rotated_bottom_left_coordinate_bb]).astype(np.int32).reshape(-1, 1, 2)], isClosed = True , color=track_colors[int(j) % LEN_OF_TRACK_COLORS], thickness=8)
                        cv2.putText(lidar_point_visualization_this_frame, "ID: " + str(tracker.tracks[j].trackId), (x - 10, y - 20), 0, 0.5, track_colors[j % LEN_OF_TRACK_COLORS], 2)
                
            plt.imsave(FOLDER_FOR_3D_BOUNDING_BOX_TRACKING_EVALUATION + "Ground_Truth_Detection_result_{}_{}.png".format(str(lidar_index_scene).zfill(4), str(index_lidar_frame).zfill(6)), cv2.cvtColor(lidar_point_visualization_this_frame, cv2.COLOR_BGR2RGB))
            video_lidar_prediction.write( lidar_point_visualization_this_frame[ : 700 , : ])
            """
    cv2.destroyAllWindows()
    video_lidar_prediction.release()
    video_image_prediction.release()

if __name__ == '__main__':
    main()
