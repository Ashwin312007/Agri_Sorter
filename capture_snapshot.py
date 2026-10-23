import cv2
import os
import shutil
import onnxruntime as ort
import detect

artifact_dir = r"C:\Users\teash\.gemini\antigravity-cli\brain\9ea35e48-330d-4b58-ae5d-604b24e7ff6b"
os.makedirs(artifact_dir, exist_ok=True)

# 1. Copy sample defect detections from saved/
samples = [
    ("images_0001275_jpg.rf.2131e30b9dfa3059fc28c91265de56cb_12.jpg", "sample_fresh.jpg"),
    ("images_0001117_jpg.rf.d370d5c217df1809118927c7641e68d6_9.jpg", "sample_bruised.jpg"),
    ("images_0001735_jpg.rf.fe54386d3212a604ee53b67e00ccb040_2.jpg", "sample_damaged.jpg"),
    ("images_0001623_jpg.rf.3db7fdb5b8a96565ea3eeba53a7fe9a7_4.jpg", "sample_rotten.jpg"),
]

for src_name, dst_name in samples:
    src_path = os.path.join("saved", src_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(artifact_dir, dst_name))
        print(f"Copied {dst_name}")

# 2. Grab live camera frame and run YOLO inference
cap = detect.open_cam(0, 640, 480)
if cap:
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if ok and frame is not None:
        raw_path = os.path.join(artifact_dir, "live_raw_cam.jpg")
        cv2.imwrite(raw_path, frame)
        print("Saved raw frame:", raw_path)

        so = ort.SessionOptions()
        sess = ort.InferenceSession("best.onnx", so, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        names = detect.load_names(sess, None, detect.APPLE_CLASSES)
        lb, r, pad = detect.letterbox(frame, 416)
        out = sess.run(None, {inp_name: detect.preprocess(lb)})[0]
        dets = detect.postprocess(out, names, r, pad, frame.shape[:2], 0.25, 0.45)

        vis = frame.copy()
        verdict = "REJECT" if any(detect.is_defect(d["name"]) for d in dets) else "FRESH"
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            col = (0, 0, 255) if detect.is_defect(d["name"]) else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{d['name']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        vcol = (0, 0, 255) if verdict == "REJECT" else (0, 200, 0)
        cv2.putText(vis, f"{verdict}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, vcol, 2)

        model_path = os.path.join(artifact_dir, "live_yolo_model.jpg")
        cv2.imwrite(model_path, vis)
        print("Saved YOLO inference frame:", model_path, "| Verdict:", verdict, "| Detections:", len(dets))
    cap.release()
print("Snapshot capture complete.")
