from ultralytics import YOLO

def predict(img):
    model = YOLO("best2.pt")
    results = model(img)

    for result in results:
        boxes = result.boxes
    
    gt = []
    for box in boxes:
        x, y, w, h = box.xywh.flatten()
        prob = box.conf
        
        x = x.item()
        y = y.item()
        w = w.item()
        h = h.item()
        prob = prob.item()

        gt += [(prob, x, y, w, h)]

        # with open("../examples1/results/nlvnpf-0137-01-011.txt", "a") as f:
        #     f.write(f"{prob} {x} {y} {w} {h}\n")
    
    return gt
    # results = model("../examples1/images/nlvnpf-0137-01-011.jpg")