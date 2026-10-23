import cv2
import glob
import onnxruntime as ort
import detect

sess = ort.InferenceSession("best_mango.onnx", providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
names = detect.load_names(sess, None, detect.MANGO_CLASSES)

test_imgs = glob.glob("dataset_mango/images/test/*.jpg")[:15]

for p in test_imgs:
    img = cv2.imread(p)
    if img is None:
        continue
    lb, r, pad = detect.letterbox(img, 416)
    out = sess.run(None, {inp_name: detect.preprocess(lb)})[0]

    dets_05 = detect.postprocess(out, names, r, pad, img.shape[:2], conf_t=0.05, iou_t=0.45)
    dets_15 = detect.postprocess(out, names, r, pad, img.shape[:2], conf_t=0.15, iou_t=0.45)
    dets_25 = detect.postprocess(out, names, r, pad, img.shape[:2], conf_t=0.25, iou_t=0.45)

    res_05 = [f"{d['name']}:{d['conf']:.2f}" for d in dets_05]
    res_15 = [f"{d['name']}:{d['conf']:.2f}" for d in dets_15]
    res_25 = [f"{d['name']}:{d['conf']:.2f}" for d in dets_25]
    print(f"File: {p[-30:]}")
    print(f"  conf=0.05: {res_05}")
    print(f"  conf=0.15: {res_15}")
    print(f"  conf=0.25: {res_25}")
