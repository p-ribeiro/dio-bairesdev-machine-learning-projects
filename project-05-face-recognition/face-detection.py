from deepface import DeepFace
import cv2
from pathlib import Path

curr_path = Path(__file__).parent
db_path = Path.joinpath(curr_path, "db")
test_images_dir = curr_path / "friends"

def identify_faces():
    
    for test_img_path in test_images_dir.iterdir():
   
        real_names = {
            "courteney": "Courteney Cox",
            "jennifer": "Jennifer Aniston",
            "lisa": "Lisa Kudrow",
            "matt": "Matt LeBlanc",
            "david": "David Schwimmer",
            "matthew": "Matthew Perry"
        }

        img = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # type: ignore

        faces = DeepFace.extract_faces(
            img_path=str(test_img_path),
            detector_backend="yolov11m",
            enforce_detection=False,
            align=False
        )
        
        
        for face in faces:
            area = face["facial_area"]
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        
            face_img = (face["face"] * 255).astype('uint8')
            
            temp_face_img = Path.joinpath(curr_path, "temp_face.png")
            cv2.imwrite(str(temp_face_img), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            result = DeepFace.find(
                img_path=str(temp_face_img),
                db_path=str(db_path),
                enforce_detection=False,
                detector_backend="yolov11m",
                model_name="VGG-Face",
                distance_metric="cosine",
            )
            
            if not result[0].empty: # type: ignore
                best_match = result[0].iloc[0] # type: ignore
                name = real_names.get(Path(best_match['identity']).stem[:-2], "Unknown")
                
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (204, 0, 0), 2)
                cv2.putText(img_rgb, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1.2, (204, 0, 0), 2)
        
        result_path = Path.joinpath(curr_path, "results", test_img_path.name)
        cv2.imwrite(str(result_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

identify_faces()






