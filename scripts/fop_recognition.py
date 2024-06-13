import cv2
import torch
import time
import os
import uuid
import sqlalchemy
import csv
import json

from ultralytics import YOLO
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from db_models.models import AdditionalData
from scripts.config import settings


def distanse_from_ebed(
      connection: sqlalchemy.engine.base.Connection,
      person_embed,
      face_embed,
      person_trashhold: float,
      face_trashhold: float
) -> list:
    """
    func using to check the presence of vectors in the database 
    that are close according to the specified values cosine distance - person_trashhold and face_trashhold
    """
    cosine_distance_to_person_embedding = AdditionalData.person_vector.cosine_distance(person_embed[0])
    cosine_distance_to_face_embedding = AdditionalData.face_vector.cosine_distance(face_embed[0])

    results = connection.execute(select(
      *AdditionalData.__table__.c,
      cosine_distance_to_person_embedding.label('person_cosine_distance'),
      cosine_distance_to_face_embedding.label('face_cosine_distance')
      ).select_from(AdditionalData).order_by(cosine_distance_to_face_embedding).where(cosine_distance_to_person_embedding < person_trashhold).where(cosine_distance_to_face_embedding < face_trashhold))

    return results.fetchall()

class FOPRecognition:
    """
    class for face of person recognition
    """
    def __init__(
            self,
            person_model_path: str = 'dploma_face_recognition/ml_models/yolov8n.pt',
            face_model_path: str = 'dploma_face_recognition/ml_models/yolov8n-face.pt',
            save_image: bool = False,
            write_video: bool = False,
            results_file: bool = False,
            person_trashhold: float = 0.1,
            face_trashhold: float = 0.1,
            model_conf: float = 0.25,
            face_model_conf: float = 0.5
            ):
        self.person_model_path = person_model_path
        self.face_model_path = face_model_path
        self.save_image = save_image
        self.write_video = write_video
        self.results_file = results_file
        self.person_trashhold = person_trashhold
        self.face_trashhold = face_trashhold
        self.model_conf = model_conf
        self.face_model_conf = face_model_conf

    def face_recogniton(
            self, 
            video_path: str, 
            crop_dir_name: str,
            video_write_dir_name: str,
            results_file_dir_name: str
            ) -> json:
        
        # init model 2 for recognition 2 for conver pic to vec
        person_model = YOLO(self.person_model_path)
        embed_person_model = YOLO(self.face_model_path)
        face_model = YOLO(self.person_model_path)
        embed_face_model = YOLO(self.face_model_path)
        names = person_model.names

        a, b, c, d, e, f, g, k = 0, 0, 0, 0, 0, 0, 0, 0

        start_time = time.time()

        # Open video and check videofile openning
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), 'Error reading video file'
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Create dir for save image if it's not
        if self.save_image:
            if not os.path.exists(crop_dir_name):
                os.mkdir(crop_dir_name)
            if not os.path.exists(crop_dir_name + '/face'):
                os.mkdir(crop_dir_name + '/face')

        # Create video writer
        if self.write_video:
            video_writer = cv2.VideoWriter(
                video_write_dir_name,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, 
                (w, h)
                )

        # init index for image, dict for storage ifo about detective person
        idx = 0
        person_face_dict = {}

        # init db connection
        engine = create_engine(
           settings.DATABASE_URI,
           isolation_level='SERIALIZABLE'
           )
        
        # check and write type of device for processing
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(device))

        with engine.connect() as connection:
            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print('Video frame is empty or video processing has been successfully completed.')
                    break    
                
                a += 1

                # tracking person on image 
                results = person_model.track(
                    im0,
                    classes=[0],
                    conf=self.model_conf,
                    persist=True,
                    device=device,
                    show=False
                    )
                # send results data to videocard
                try:
                    person_boxes = results[0].boxes.xyxy.cuda().tolist()
                    person_id = results[0].boxes.id.cuda().tolist()
                    annotator = Annotator(im0, line_width=1, example=names)
                except AttributeError:
                    person_id = []
                if person_boxes is not None:
                    for box, id in zip(person_boxes, person_id):
              
                        idx += 1
                        crop_obj = im0[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
              
                        face_results = face_model.predict(
                            crop_obj,
                            classes=[0],
                            conf=self.face_model_conf,
                            device=device,
                            show=False
                            )
              
                        annotator.box_label(box, color=colors(0, True), label=f'person_{int(id)}')
              
                        if face_results[0].verbose()[-17:-2] == '(no detections)':
                            continue

                        elif (id not in list(person_face_dict.keys())):
                            d += 1
                            c += 1
                            # get coordinates of boxes with face
                            face_boxes = face_results[0].boxes.xyxy.cuda().tolist()

                            crop_face = im0[int(box[1])+int(face_boxes[0][1]):int(box[1])+int(face_boxes[0][3]),
                                                int(box[0])+int(face_boxes[0][0]):int(box[0])+int(face_boxes[0][2])]

                            person_embed = embed_person_model.embed(crop_obj)
                            face_embed = embed_face_model.embed(crop_face)

                            # printing boxes on image
                            annotator.box_label([int(box[0])+int(face_boxes[0][0]), int(box[1])+int(face_boxes[0][1]), int(box[0])+int(face_boxes[0][2]), int(box[1])+int(face_boxes[0][3])], color=colors(5, True), label=f'person_{int(id)}_face')
                            
                            results = distanse_from_ebed(
                                connection,
                                person_embed,
                                face_embed,
                                self.person_trashhold,
                                self.face_trashhold
                                )

                            if len(results) == 0:
                                e += 1 
                                person_face_dict[id] = {
                                    'id': uuid.uuid4(),
                                    'person_id': uuid.uuid1(),
                                    'person_vector': person_embed[0],
                                    'face_vector': face_embed[0],
                                    'face_detection_conf': float(face_results[0].boxes.conf[0])
                                }
                                cv2.imwrite(os.path.join(crop_dir_name, f'new_person_{person_face_dict[id]["person_id"]}_{id}.png'), crop_obj)
                                cv2.imwrite(os.path.join(crop_dir_name + '/face', f'new_face_person_{person_face_dict[id]["person_id"]}_{id}_{face_results[0].boxes.conf[0]}.png'), crop_face)
                                
                                connection.execute(insert(AdditionalData).values(person_face_dict[id]).on_conflict_do_update(
                                    index_elements=['id'],
                                    set_={
                                        'person_id': person_face_dict[id]['person_id'],
                                        'person_vector': person_face_dict[id]['person_vector'],
                                        'face_vector': person_face_dict[id]['face_vector'],
                                        'face_detection_conf': person_face_dict[id]['face_detection_conf']
                                        }
                                        ))
                                connection.commit()

                            elif face_results[0].boxes.conf[0] > results[0][5]:
                                f += 1
                                person_face_dict[id] = {
                                    'id': results[0][1],
                                    'person_id': results[0][2],
                                    'person_vector': person_embed[0],
                                    'face_vector': face_embed[0],
                                    'face_detection_conf': float(face_results[0].boxes.conf[0])
                                }
                                cv2.imwrite(os.path.join(crop_dir_name, f'old_person_{person_face_dict[id]["person_id"]}_{id}_{person_face_dict[id]["person_id"]}.png'), crop_obj)
                                cv2.imwrite(os.path.join(crop_dir_name + '/face', f'old_face_person_{person_face_dict[id]["person_id"]}_{id}_{face_results[0].boxes.conf[0]}.png'), crop_face)
                            else:
                                continue   
                            
                            print(f'ADDED with id - {int(id)}, conf - {face_results[0].boxes.conf[0]}')
              
                        elif ((id in list(person_face_dict.keys())) and (face_results[0].boxes.conf[0] > person_face_dict[id]['face_detection_conf'])):
                            k += 1
                            c += 1
                            person_face_dict[id] = {
                            'id': person_face_dict[id]['id'],
                            'person_id': person_face_dict[id]['person_id'],
                            'person_vector': person_embed[0],
                            'face_vector': face_embed[0],
                            'face_detection_conf': float(face_results[0].boxes.conf[0])
                            }

                        else:
                            continue
                
                # show writing video
                if self.write_video:
                    cv2.imshow('ultralytics', im0)
                    video_writer.write(im0)
                    if (cv2.waitKey(1) & 0xFF == ord('q')):
                        break
            
            # write new detected perdon or rewrite previously detected person
            for idx in list(person_face_dict.keys()):
                g += 1
                connection.execute(insert(AdditionalData).values(person_face_dict[idx]).on_conflict_do_update(
                            index_elements=['id'],
                            set_={
                                    'person_id': person_face_dict[idx]['person_id'],
                                    'person_vector': person_face_dict[idx]['person_vector'],
                                    'face_vector': person_face_dict[idx]['face_vector'],
                                    'face_detection_conf': person_face_dict[idx]['face_detection_conf']
                                    }
                            ))

            connection.commit()

            # Video writer    
            if self.write_video:
                cap.release()
                video_writer.release()
                cv2.destroyAllWindows()
 
        end_time = time.time()  
        execution_time = end_time - start_time 

        if self.results_file:
            with open(results_file_dir_name + '/results.csv', 'a', newline='') as csvfile:
                fieldnames = ['all_img', 'img_with_face', 'img_new_person', 'img_face_more_conf', 'new_wr', 'rewrite', 'execution_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerow({'all_img': a, 
                                'img_with_face': c, 
                                'img_new_person': d, 
                                'img_face_more_conf': k, 
                                'new_wr': e, 
                                'rewrite': f,
                                'execution_time': execution_time})
        
        return json.dumps({
            'all_img': a,
            'img_with_face': c,
            'img_new_person': d,
            'img_face_more_conf': k,
            'new_db_write': e,
            'db_rewrite': f,
            'execution_time': execution_time
            })